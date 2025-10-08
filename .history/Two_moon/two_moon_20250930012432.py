import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import math
from tqdm import tqdm
from typing import Dict
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
import imageio


# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Experiment parameters
N_SAMPLES_TRAIN = 200
N_SAMPLES_TEST = 1000
INPUT_DIM = 2
HIDDEN_DIM = 32
OUTPUT_DIM = 2
MAX_EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-2
SEED = 100

# DRO parameters
EPSILON = 0.01
LAMBDA_PARAM = 5
NUM_SAMPLES_PER_POINT = 5
SINKHORN_SAMPLE_LEVEL = 4
inner_lr = 1e-2
inner_steps = 200

MMD_NOISE_STD = 1e-3
wfr_times = 16

OUTPUT_DIR = "two_moons_results"
OUTPUT_DIR_PDF = "two_moons_results_pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_PDF, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 2. Model and Data ---
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def create_dataset(n_samples=500, noise=0.15, imbalance_ratio=0.5, random_state=42):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples*2, noise=noise, random_state=random_state)
    X_pos, y_pos, X_neg, y_neg = X[y == 1], y[y == 1], X[y == 0], y[y == 0]
    n_pos, n_neg = int(n_samples * imbalance_ratio), int(n_samples * (1-imbalance_ratio))
    X_imbalanced = np.vstack([X_pos[:n_pos], X_neg[:n_neg]])
    y_imbalanced = np.hstack([y_pos[:n_pos], y_neg[:n_neg]])
    shuffle_idx = np.random.permutation(len(X_imbalanced))
    return X_imbalanced[shuffle_idx], y_imbalanced[shuffle_idx]

def imq_kernel_and_grad(X, Y, c=1.0):
    m, d = X.shape
    n, _ = Y.shape

    X_sq = torch.sum(X**2, dim=1, keepdim=True)
    Y_sq = torch.sum(Y**2, dim=1, keepdim=True)
    XY = torch.matmul(X, Y.T)
    dist_sq = (X_sq - 2*XY + Y_sq.T).clamp(min=0)

    K = c / torch.sqrt(c**2 + dist_sq)

    diff = X.unsqueeze(1).expand(m, n, d) - Y.unsqueeze(0).expand(m, n, d)

    factor = -(c**2 + dist_sq)**(-1.5)
    grad_K = factor.unsqueeze(2) * diff *c

    return K, grad_K

def rbf_kernel_and_grad(X, Y):
    m, d = X.shape
    n, _ = Y.shape

    X_sq = torch.sum(X**2, dim=1, keepdim=True)
    Y_sq = torch.sum(Y**2, dim=1, keepdim=True)
    XY = torch.matmul(X, Y.T)
    dist_sq = (X_sq - 2*XY + Y_sq.T).clamp(min=0)

    X_dist_sq = (X_sq - 2 * torch.matmul(X, X.T) + X_sq.T).clamp(min=0)
    all_dist_sq_x = X_dist_sq.flatten()

    if all_dist_sq_x.numel() > 1:
        h_sq = 0.5 * torch.median(all_dist_sq_x) / torch.log(torch.tensor(m + 1.0, device=X.device))
    else:
        h_sq = torch.tensor(1.0, device=X.device)
    h_sq = torch.clamp(h_sq, min=1e-6)

    K = torch.exp(-dist_sq / (2 * h_sq))

    diff = X.unsqueeze(1).expand(m, n, d) - Y.unsqueeze(0).expand(m, n, d)
    factor = -K / h_sq
    grad_K = factor.unsqueeze(2) * diff

    return K, grad_K

def mmd_dro_sampler(x_original_batch, y_original_batch, model, epoch, kernel_fn=imq_kernel_and_grad, lr=0.01, inner_steps=100):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    x_clone = x_clone.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)


    current_inner_steps = int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))
    for _ in range(current_inner_steps):
        x_clone.requires_grad_(True)
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
        loss_grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone.detach()

        _, grad_K_clone = kernel_fn(x_clone, x_clone)
        mmd_term1 = torch.mean(grad_K_clone, dim=1) 

        _, grad_K_orig = kernel_fn(x_clone, x_original_batch)
        mmd_term2 = torch.mean(grad_K_orig, dim=1) 

        velocity = loss_grads - LAMBDA_PARAM * (mmd_term1 - mmd_term2)

        noise = torch.randn_like(x_clone) * MMD_NOISE_STD
        
        x_clone = x_clone + lr * velocity + noise

    return x_clone.detach()

def wgf_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    x_clone = x_clone.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).reshape(-1, INPUT_DIM)
    for _ in range(int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))):
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        mean = x_clone +  lr * (grads  - 2*LAMBDA_PARAM*(x_clone - x_original_expanded))
        std_dev = torch.sqrt(torch.tensor(2*lr*LAMBDA_PARAM*EPSILON, device=DEVICE))
        mean_expanded = mean.unsqueeze(1).expand(-1, 1, -1)
        noise = torch.randn_like(mean_expanded) * std_dev
        x_clone = (mean_expanded + noise).view(-1, INPUT_DIM)
    return x_clone.detach()

def wfr_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    device = x_original_batch.device
    batch_size = x_original_batch.shape[0]

    weights = torch.ones((batch_size, NUM_SAMPLES_PER_POINT), dtype=torch.float32, device=device) / NUM_SAMPLES_PER_POINT

    x_clone = x_original_batch.clone().detach().unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).reshape(-1, INPUT_DIM)

    wt_lr = lr * wfr_times
    weight_exponent = 1 - LAMBDA_PARAM * EPSILON * wt_lr

    for _ in range(int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))):
        x_clone.requires_grad_(True)

        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone.detach()

        mean = x_clone + lr * (grads - 2 * LAMBDA_PARAM * (x_clone - x_original_expanded))
        std_dev = torch.sqrt(torch.tensor(2*lr*LAMBDA_PARAM*EPSILON, device=device))

        mean_expanded = mean.unsqueeze(1).expand(-1, 1, -1)
        noise = torch.randn_like(mean_expanded) * std_dev
        x_clone = (mean_expanded + noise).view(-1, INPUT_DIM)

        with torch.no_grad():
            new_loss = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated).view(batch_size, -1) - 2 * LAMBDA_PARAM * torch.sum((x_clone - x_original_expanded) ** 2, dim=1).view(batch_size, -1)

            weights = weights ** weight_exponent * torch.exp(new_loss * wt_lr)

            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

        with torch.no_grad():
            low_weight_mask = weights < 1e-4
            rows_with_low_weights = torch.any(low_weight_mask, dim=1)

            if torch.any(rows_with_low_weights):

                x_reshaped = x_clone.view(batch_size, NUM_SAMPLES_PER_POINT, -1)

                max_weight_vals, max_weight_indices = torch.max(weights, dim=1, keepdim=True)

                highest_weight_point_data = torch.gather(x_reshaped, 1, max_weight_indices.unsqueeze(-1).expand(-1, -1, INPUT_DIM))

                low_weights_sum = torch.sum(weights * low_weight_mask, dim=1, keepdim=True)
                num_low_weights = torch.sum(low_weight_mask, dim=1, keepdim=True, dtype=weights.dtype)

                avg_weight = (max_weight_vals + low_weights_sum) / (num_low_weights + 1.0 + 1e-9)
                avg_weight_expanded = avg_weight.expand_as(weights)

                max_weight_mask = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, max_weight_indices, True)

                update_mask = (low_weight_mask | max_weight_mask) & rows_with_low_weights.unsqueeze(1)
                weights = torch.where(update_mask, avg_weight_expanded, weights)

                x_update_mask = low_weight_mask & rows_with_low_weights.unsqueeze(1)
                replacement_data = highest_weight_point_data.expand_as(x_reshaped)
                x_reshaped = torch.where(x_update_mask.unsqueeze(-1), replacement_data, x_reshaped)
                
                x_clone = x_reshaped.view(-1, INPUT_DIM)

                weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

    return x_clone.detach(), weights.detach()

def wrm_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    for _ in range(int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))):
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_original_batch)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone +  lr * (grads  - 2*LAMBDA_PARAM*(x_clone - x_original_batch))
    return x_clone.detach()

def rbf_kernel_batched_torch(particles):
    B, S, D = particles.shape
    device = particles.device

    sq_dist = torch.cdist(particles, particles, p=2).pow(2)

    median_sq_dist = torch.median(sq_dist.view(B, -1), dim=1, keepdim=True)[0]
    median_sq_dist = median_sq_dist.view(B, 1, 1)

    h_squared =  median_sq_dist / (torch.log(torch.tensor(S, device=device)) + 1e-8) + 1e-6

    K = torch.exp(-sq_dist / (2 * h_squared))

    diff = particles.unsqueeze(2) - particles.unsqueeze(1)
    grad_K_x = -diff / h_squared.unsqueeze(-1) * K.unsqueeze(-1)

    return K, grad_K_x

def svgd_sampler(
    x_original_batch,
    y_original_batch,
    model,
    num_samples_per_point=NUM_SAMPLES_PER_POINT,
    lr=inner_lr,
    inner_steps=inner_steps,
    adagrad_hist_decay=0.9
):

    if x_original_batch.shape[0] == 0:
        return torch.empty(0, x_original_batch.shape[1], device=x_original_batch.device)
    device = x_original_batch.device
    B, D = x_original_batch.shape
    S = num_samples_per_point

    x_orig_repeated = x_original_batch.unsqueeze(1).repeat(1, S, 1)
    y_repeated = y_original_batch.repeat_interleave(S)

    particles = x_orig_repeated.clone().detach()
    particles += torch.randn_like(particles) * np.sqrt(EPSILON / 2)
    hist_grad = torch.zeros_like(particles)

    for _ in range(inner_steps):
        x_tensor = particles.view(B * S, D).clone().requires_grad_(True)
        
        x_orig_repeated_flat = x_orig_repeated.view(B * S, D)

        neg_log_likelihood = nn.CrossEntropyLoss(reduction='sum')(model(x_tensor), y_repeated)
        grad_log_py_x, = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_tensor)
        grad_log_px = -2 * LAMBDA_PARAM * (x_tensor - x_orig_repeated_flat)

        total_grad_flat = (grad_log_py_x + grad_log_px)/ (LAMBDA_PARAM * EPSILON)
        total_grad = total_grad_flat.view(B, S, D)

        K, grad_K_x = rbf_kernel_batched_torch(particles)
        
        K_grad_prod = torch.bmm(K, total_grad)
        
        sum_grad_K = torch.sum(grad_K_x, dim=2)

        svgd_grad = (K_grad_prod + sum_grad_K) / S

        with torch.no_grad():
            hist_grad = adagrad_hist_decay * hist_grad + (1 - adagrad_hist_decay) * (svgd_grad**2)
            adj_grad = svgd_grad / (1e-6 + torch.sqrt(hist_grad))
            particles += lr * adj_grad

    return particles.view(B * S, D).detach()

def sinkhorn_base_sampler(x_original_batch, m_samples):
    expanded_data = x_original_batch.repeat_interleave(m_samples, dim=0)
    noise = torch.randn_like(expanded_data) * math.sqrt(EPSILON)
    return expanded_data + noise

def compute_sinkhorn_loss(predictions, targets, m, lambda_reg):
    criterion = nn.CrossEntropyLoss(reduction='none')
    residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
    residual_matrix = residuals.view(-1, m).T
    return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg

# --- 4. Visualization and Evaluation ---
def plot_frame(model, X, y, X_perturbed, title, save_path, method, epoch):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5, s=15, linewidths=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5, s=15, linewidths=0.5)
    if X_perturbed is not None:
        num_repeats = X_perturbed.shape[0] // X.shape[0]
        y_repeated = np.repeat(y, num_repeats)
        X_p_np = X_perturbed.cpu().numpy()
        sns.kdeplot(x=X_p_np[y_repeated==1, 0], y=X_p_np[y_repeated==1, 1], ax=ax, color='darkorange', fill=True, alpha=0.4)
        sns.kdeplot(x=X_p_np[y_repeated==0, 0], y=X_p_np[y_repeated==0, 1], ax=ax, color='dodgerblue', fill=True, alpha=0.4)
    x_min, x_max, y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1, X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        Z_model = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    ax.contour(xx, yy, Z_model, levels=[0.5], linewidths=1.5, colors='black', linestyles='--')
    legend_elements = [Line2D([0], [0], color='black', linestyle='--', lw=3, label=f'Boundary({method})'),
                       Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=4, markeredgecolor='k', markeredgewidth=0.5),
                       Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=4, markeredgecolor='k', markeredgewidth=0.5),
                       Patch(facecolor='darkorange', alpha=0.4, label='Positive Worst-Case Dist.'),
                       Patch(facecolor='dodgerblue', alpha=0.4, label='Negative Worst-Case Dist.')]
    ax.set_xlabel('feature 1', fontsize=9); ax.set_ylabel('feature 2', fontsize=9)
    ax.text(0.95, 0.05, f'Epoch: {epoch}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7); #ax.set_title(title, fontsize=9); 
    ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    plt.savefig(save_path); plt.close(fig)

def plot_frame_samples(model, X, y, X_perturbed, title, save_path, method, epoch, weights=None):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)

    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5, s=20, linewidths=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5, s=20, linewidths=0.5)

    if X_perturbed is not None:
        X_p_np = X_perturbed.cpu().numpy()
        num_repeats = X_perturbed.shape[0] // X.shape[0]
        y_repeated = np.repeat(y, num_repeats)

        if weights is None:
            weights = torch.ones((X.shape[0], num_repeats), device=X_perturbed.device) / num_repeats

        weights_flat = weights.flatten().cpu().numpy()

        min_alpha, max_alpha = 0, 0.5
        w_min, w_max = weights_flat.min(), weights_flat.max()
        if w_max > w_min:
            alphas = min_alpha + (weights_flat - w_min) / (w_max - w_min) * (max_alpha - min_alpha)
        else: 
            alphas = np.full_like(weights_flat, max_alpha/2)

        colors_rgba = np.zeros((X_p_np.shape[0], 4))
        color_pos_rgb = to_rgb('darkorange')
        color_neg_rgb = to_rgb('dodgerblue')

        pos_mask = y_repeated == 1
        neg_mask = y_repeated == 0

        colors_rgba[pos_mask, :3] = color_pos_rgb
        colors_rgba[neg_mask, :3] = color_neg_rgb
        colors_rgba[:, 3] = alphas 

        ax.scatter(X_p_np[:, 0], X_p_np[:, 1], c=colors_rgba, marker='o',edgecolors="gray", linewidths=0.2, s=15)

        for i in range(X.shape[0]):
            original_point = X[i]
            start_index = i * num_repeats
            end_index = start_index + num_repeats
            perturbed_points_block = X_p_np[start_index:end_index, :]

            for p_point in perturbed_points_block:
                ax.plot([original_point[0], p_point[0]],
                        [original_point[1], p_point[1]],
                        color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        Z_model = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)

    ax.contour(xx, yy, Z_model, levels=[0.5], linewidths=1.5, colors='black', linestyles='--')

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', lw=2, label=f'Boundary ({method})'),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Perturbed Positive', markersize=5, markeredgecolor="gray", markeredgewidth=0.2),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Perturbed Negative', markersize=5, markeredgecolor="gray", markeredgewidth=0.2)
    ]

    ax.set_xlabel('feature 1', fontsize=9)
    ax.set_ylabel('feature 2', fontsize=9)
    if isinstance(epoch, int):
        epoch_text = f'Epoch: {epoch}'
    else: 
        epoch_text = epoch
    # ax.text(0.95, 0.05, epoch_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    #ax.set_title(title, fontsize=9)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    plt.savefig(save_path)
    
    save_path_pdf = save_path.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_pdf)
    
    plt.close(fig)

def visualize_all_boundaries(models: Dict[str, nn.Module], X, y, title, save_path_png):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5, linewidths=0.5, s=20)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5, linewidths=0.5, s=20)
    x_min, x_max, y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1, X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    colors = [
        "#E2E60C", 
        "#C9151E", 
        "#3A1FB4",
        '#CC6677', 
        "#951699",   
        "#1C7414EF",
        "#FF7F00"
    ]
    linestyles = ['--', ':', '-.', '--']

    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linewidths=1.5)

    ideal_color = 'black'
    ideal_linewidth = 1.5
    center1 = (0, 0.25)
    center2 = (1, 0.25)
    radius = 0.5

    arc1 = Arc(center1, width=radius*2, height=radius*2, theta1=0, theta2=180,
               edgecolor=ideal_color, lw=ideal_linewidth, linestyle='-')
    ax.add_patch(arc1)

    arc2 = Arc(center2, width=radius*2, height=radius*2, theta1=180, theta2=360,
               edgecolor=ideal_color, lw=ideal_linewidth, linestyle='-')
    ax.add_patch(arc2)


    ax.plot([-0.5, -0.5], [0.25, y_min], color=ideal_color, lw=ideal_linewidth, linestyle='-')

    ax.plot([1.5, 1.5], [0.25, y_max], color=ideal_color, lw=ideal_linewidth, linestyle='-')

    legend_elements = [Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=4, markeredgecolor='k', markeredgewidth=0.5),
                       Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=4, markeredgecolor='k', markeredgewidth=0.5),]
    legend_elements.extend([Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="Ideal boundary")])
    legend_elements.extend([Line2D([0], [0], lw=1.5, color=colors[i], label=name) for i, name in enumerate(models.keys())])
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1)) #; ax.set_title(title)
    ax.set_xlabel('feature 1'); ax.set_ylabel('feature 2'); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    
    save_path_pdf = save_path_png.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    plt.close(fig)

def plot_loss_comparison(loss_histories: Dict[str, list], save_path_png):
    plt.figure(figsize=(4, 2.613), dpi=300, tight_layout=True)
    for name, history in loss_histories.items():
        epochs_logged = range(1, (len(history) * LOG_EPOCH_INTERVAL) + 1, LOG_EPOCH_INTERVAL)
        plt.plot(epochs_logged, history, label=name, marker='o',  markersize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    #plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    
    save_path_pdf = save_path_png.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    plt.close()

def plot_frame_samples_with_forces(model, X, y, X_perturbed, title, save_path, method, epoch, force1, force2, weights=None):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)

    # 1. Plot original data points
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5, s=20, linewidths=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5, s=20, linewidths=0.5)

    if X_perturbed is not None:
        X_p_np = X_perturbed.cpu().numpy()
        num_repeats = X_perturbed.shape[0] // X.shape[0]
        y_repeated = np.repeat(y, num_repeats)

        # Plot perturbed points
        colors_rgba = np.zeros((X_p_np.shape[0], 4))
        colors_rgba[y_repeated == 1, :3] = to_rgb('darkorange')
        colors_rgba[y_repeated == 0, :3] = to_rgb('dodgerblue')
        colors_rgba[:, 3] = 0.5
        ax.scatter(X_p_np[:, 0], X_p_np[:, 1], c=colors_rgba, marker='o',edgecolors="gray", linewidths=0.2, s=15)

        # Plot forces with ax.quiver
        force1_np = force1.detach().cpu().numpy()
        force2_np = force2.detach().cpu().numpy()
        arrow_scale = 0.3
        ax.quiver(X_p_np[:, 0], X_p_np[:, 1], force1_np[:, 0], force1_np[:, 1],
                  color='lightgray', alpha=0.35, scale_units='xy', angles='xy', scale=1/arrow_scale,
                  width=0.005, headwidth=3)
        ax.quiver(X_p_np[:, 0], X_p_np[:, 1], force2_np[:, 0], force2_np[:, 1],
                  color='khaki', alpha=0.35, scale_units='xy', angles='xy', scale=1/arrow_scale,
                  width=0.005, headwidth=3)
                  
    # 3. Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        Z_model = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    ax.contour(xx, yy, Z_model, levels=[0.5], linewidths=1.5, colors='black', linestyles='--')

    # 4. Create legend with dynamic labels based on the method
    force_labels = {
        'WGF': (r'$\nabla_z f_\theta$', r'$-2\lambda(z-x)$'),
        'WFR': (r'$\nabla_z f_\theta$', r'$-2\lambda(z-x)$'),
        'WRM': (r'$\nabla_z f_\theta$', r'$-2\lambda(z-x)$'),
        'MMD': (r'$\nabla_z f_\theta$', r'kernel grad'),
        'SVGD': (r'loss kernel', r'kernel grad')
    }
    force1_label, force2_label = force_labels.get(method, ('Force 1', 'Force 2'))

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', lw=2, label=f'Boundary (SAA)'),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Patch(facecolor='lightgray', alpha=0.9, label=force1_label),
        Patch(facecolor='khaki', alpha=0.9, label=force2_label)
    ]

    # 5. Set plot attributes and save
    ax.set_xlabel('feature 1', fontsize=9)
    ax.set_ylabel('feature 2', fontsize=9)
    # ax.text(0.95, 0.05, epoch, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    #ax.set_title(title, fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.savefig(save_path)
    save_path_pdf = save_path.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_pdf)
    plt.close(fig)

def plot_perturbation_process_wasserstein_forces(saa_model, X_train, y_train, method, output_dir, lr=0.01, inner_steps=200):
    print(f"\n--- Visualizing SAA Perturbation Process for {method} with Forces ---")
    frames_dir = os.path.join(output_dir, f"saa_{method}_forces_frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames_dir_pdf = os.path.join(OUTPUT_DIR_PDF, f"saa_{method}_forces_frames")
    os.makedirs(frames_dir_pdf, exist_ok=True)
    
    gif_filenames = []
    objective_history = []

    device = next(saa_model.parameters()).device
    saa_model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    X_train_torch = torch.from_numpy(X_train).float().to(device)
    y_train_torch = torch.from_numpy(y_train).long().to(device)
    batch_size = X_train_torch.shape[0]

    num_particles = NUM_SAMPLES_PER_POINT
    x_clone = X_train_torch.clone().detach().unsqueeze(1).expand(-1, num_particles, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = y_train_torch.repeat_interleave(num_particles, dim=0)
    x_original_expanded = X_train_torch.unsqueeze(1).expand(-1, num_particles, -1).reshape(-1, INPUT_DIM)

    weights = None
    if method == 'WFR':
        weights = torch.ones((batch_size, num_particles), dtype=torch.float32, device=device) / num_particles
        wt_lr = lr * wfr_times
        weight_exponent = 1 - LAMBDA_PARAM * EPSILON * wt_lr


    for step in tqdm(range(inner_steps), desc=f"Generating {method} force frames"):
        
        with torch.no_grad():
            loss_values = criterion(saa_model(x_clone), y_repeated)
            dist_sq = torch.sum((x_clone - x_original_expanded)**2, dim=1)
            objective_per_sample = loss_values - LAMBDA_PARAM * dist_sq
            if method == 'WFR':
                objective_reshaped = objective_per_sample.view(batch_size, -1)
                avg_objective = (objective_reshaped * weights).sum(dim=1).mean().item()
            else: 
                avg_objective = objective_per_sample.mean().item()
            objective_history.append(avg_objective)

        x_clone.requires_grad_(True)

        loss_values_for_grad = nn.CrossEntropyLoss(reduction='none')(saa_model(x_clone), y_repeated)
        force1_loss_grad, = torch.autograd.grad(loss_values_for_grad, x_clone, grad_outputs=torch.ones_like(loss_values_for_grad))
        x_clone = x_clone.detach()
        force2_anchor_grad = -2 * LAMBDA_PARAM * (x_clone - x_original_expanded)

        total_velocity = force1_loss_grad + force2_anchor_grad
        
        if method == 'WRM':
            x_clone += lr * total_velocity
        else: 
            mean = x_clone + lr * total_velocity
            std_dev = torch.sqrt(torch.tensor(2 * lr * LAMBDA_PARAM * EPSILON, device=device))
            noise = torch.randn_like(mean) * std_dev
            x_clone = mean + noise

        if method == 'WFR':
            with torch.no_grad():
                new_loss = nn.CrossEntropyLoss(reduction='none')(saa_model(x_clone), y_repeated).view(batch_size, -1) - 2 * LAMBDA_PARAM * torch.sum((x_clone - x_original_expanded) ** 2, dim=1).view(batch_size, -1)
                weights = weights ** weight_exponent * torch.exp(new_loss * wt_lr)
                weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

            with torch.no_grad():
                low_weight_mask = weights < 1e-3
                rows_with_low_weights = torch.any(low_weight_mask, dim=1)
                if torch.any(rows_with_low_weights):
                    x_reshaped = x_clone.view(batch_size, num_particles, -1)
                    max_weight_vals, max_weight_indices = torch.max(weights, dim=1, keepdim=True)
                    highest_weight_point_data = torch.gather(x_reshaped, 1, max_weight_indices.unsqueeze(-1).expand(-1, -1, INPUT_DIM))
                    low_weights_sum = torch.sum(weights * low_weight_mask, dim=1, keepdim=True)
                    num_low_weights = torch.sum(low_weight_mask, dim=1, keepdim=True, dtype=weights.dtype)
                    avg_weight = (max_weight_vals + low_weights_sum) / (num_low_weights + 1.0)
                    avg_weight_expanded = avg_weight.expand_as(weights)
                    max_weight_mask = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, max_weight_indices, True)
                    update_mask = (low_weight_mask | max_weight_mask) & rows_with_low_weights.unsqueeze(1)
                    weights = torch.where(update_mask, avg_weight_expanded, weights)
                    x_update_mask = low_weight_mask & rows_with_low_weights.unsqueeze(1)
                    replacement_data = highest_weight_point_data.expand_as(x_reshaped)
                    x_reshaped = torch.where(x_update_mask.unsqueeze(-1), replacement_data, x_reshaped)
                    x_clone = x_reshaped.view(-1, INPUT_DIM)
                    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)
        
        if step % 4 == 0:
             frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
             plot_frame_samples_with_forces(
                 model=saa_model, X=X_train, y=y_train, 
                 X_perturbed=x_clone, 
                 title=f"SAA Perturbation with {method} Forces", 
                 save_path=frame_path, method=method, epoch=f"Step: {step+1}",
                 force1=0.2*force1_loss_grad,
                 force2=0.2*force2_anchor_grad,
                 weights=weights
             )
             gif_filenames.append(frame_path)

    if imageio:
        print(f"Creating SAA {method} force visualization GIF...")
        gif_path = os.path.join(output_dir, f"saa_perturbation_{method}_forces.gif")
        with imageio.get_writer(gif_path, mode='I', duration=100, loop=0, format='gif') as writer:
            for filename in gif_filenames:
                writer.append_data(imageio.imread(filename))
        print(f"GIF saved to {gif_path}")
    
    return objective_history

def plot_perturbation_process_kernel_force(saa_model, X_train, y_train, method, output_dir, lr=0.01, inner_steps=200, adagrad_hist_decay=0.9):
    print(f"\n--- Visualizing SAA Perturbation Process for {method} with Forces ---")
    frames_dir = os.path.join(output_dir, f"saa_{method}_forces_frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames_dir_pdf = os.path.join(OUTPUT_DIR_PDF, f"saa_{method}_forces_frames")
    os.makedirs(frames_dir_pdf, exist_ok=True)
    gif_filenames, objective_history = [], []

    device = next(saa_model.parameters()).device
    saa_model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    X_train_torch = torch.from_numpy(X_train).float().to(device)
    y_train_torch = torch.from_numpy(y_train).long().to(device)
    B, D = X_train_torch.shape
    S = NUM_SAMPLES_PER_POINT
    y_repeated = y_train_torch.repeat_interleave(S)
    x_original_expanded = X_train_torch.unsqueeze(1).expand(-1, S, -1).reshape(-1, D)
    
    if method == "MMD":
        particles = x_original_expanded.clone()
        for step in tqdm(range(inner_steps), desc=f"Generating {method} force frames"):
            with torch.no_grad():
                loss_values = criterion(saa_model(particles), y_repeated)
                dist_sq = torch.sum((particles - x_original_expanded)**2, dim=1)
                avg_objective = (loss_values - LAMBDA_PARAM * dist_sq).mean().item()
                objective_history.append(avg_objective)

            particles.requires_grad_(True)
            loss_values_for_grad = criterion(saa_model(particles), y_repeated)
            force1_loss_grad, = torch.autograd.grad(loss_values_for_grad.sum(), particles)
            particles = particles.detach()

            _, grad_K_clone = imq_kernel_and_grad(particles, particles)
            mmd_term1 = torch.mean(grad_K_clone, dim=1) 
            _, grad_K_orig = imq_kernel_and_grad(particles, X_train_torch)
            mmd_term2 = torch.mean(grad_K_orig, dim=1)
            force2_kernel_grad = -LAMBDA_PARAM * (mmd_term1 - mmd_term2)

            velocity = force1_loss_grad + force2_kernel_grad
            particles += lr * velocity + torch.randn_like(particles) * MMD_NOISE_STD
            
            if step % 5 == 0:
                 frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
                 plot_frame_samples_with_forces(saa_model, X_train, y_train, particles, f"SAA Perturbation with {method} Forces", 
                                                frame_path, method, f"Step: {step+1}", 0.2 * force1_loss_grad, 0.2 * force2_kernel_grad)
                 gif_filenames.append(frame_path)

    elif method == "SVGD":
        particles = X_train_torch.unsqueeze(1).repeat(1, S, 1) + torch.randn(B, S, D, device=device) * np.sqrt(EPSILON / 2)
        hist_grad = torch.zeros_like(particles)
        for step in tqdm(range(inner_steps), desc=f"Generating {method} force frames"):
            with torch.no_grad():
                particles_flat = particles.view(-1, D)
                loss_values = criterion(saa_model(particles_flat), y_repeated)
                dist_sq = torch.sum((particles_flat - x_original_expanded)**2, dim=1)
                avg_objective = (loss_values - LAMBDA_PARAM * dist_sq).mean().item()
                objective_history.append(avg_objective)

            x_tensor_flat = particles.view(-1, D).clone().requires_grad_(True)
            loss_val = nn.CrossEntropyLoss(reduction='sum')(saa_model(x_tensor_flat), y_repeated)
            grad_log_py_x, = torch.autograd.grad(outputs=loss_val, inputs=x_tensor_flat)
            grad_log_px = -2 * LAMBDA_PARAM * (x_tensor_flat - x_original_expanded)
            total_grad_flat = grad_log_py_x + grad_log_px
            total_grad = total_grad_flat.view(B, S, D)/LAMBDA_PARAM/EPSILON

            K, grad_K_x = rbf_kernel_batched_torch(particles)
            force1_driving = torch.bmm(K, total_grad) / S
            force2_repulsive = torch.sum(grad_K_x, dim=2) / S
            svgd_grad = force1_driving + force2_repulsive
            
            with torch.no_grad():
                hist_grad = adagrad_hist_decay * hist_grad + (1 - adagrad_hist_decay) * (svgd_grad**2)
                adj_grad = svgd_grad / (1e-6 + torch.sqrt(hist_grad))
                particles += lr * adj_grad

            if step % 4 == 0:
                 frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
                 plot_frame_samples_with_forces(saa_model, X_train, y_train, particles.view(-1, D), f"SAA Perturbation with {method} Forces",
                                                frame_path, method, f"Step: {step+1}", 0.2 * LAMBDA_PARAM*EPSILON * force1_driving.view(-1,D), 0.2 * LAMBDA_PARAM*EPSILON * force2_repulsive.view(-1,D))
                 gif_filenames.append(frame_path)

    if imageio and gif_filenames:
        gif_path = os.path.join(output_dir, f"saa_perturbation_{method}_forces.gif")
        imageio.mimsave(gif_path, [imageio.imread(f) for f in gif_filenames], duration=0.1)
        print(f"GIF saved to {gif_path}")
    
    return objective_history


def plot_perturbation_process(saa_model, X_train, y_train, method, output_dir, lr=0.001, inner_steps=500, adagrad_hist_decay=0.5):
    print(f"\n--- Visualizing SAA Perturbation for {method} WITHOUT FORCES ---")
    frames_dir = os.path.join(output_dir, f"saa_{method}_no_forces_frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames_dir_pdf = frames_dir.replace(OUTPUT_DIR, OUTPUT_DIR_PDF)
    os.makedirs(frames_dir_pdf, exist_ok=True)
    gif_filenames, objective_history = [], []
    device = next(saa_model.parameters()).device
    saa_model.eval(); criterion = nn.CrossEntropyLoss(reduction='none')
    X_train_torch = torch.from_numpy(X_train).float().to(device)
    y_train_torch = torch.from_numpy(y_train).long().to(device)
    B, D, S = X_train_torch.shape[0], X_train_torch.shape[1], NUM_SAMPLES_PER_POINT
    y_repeated = y_train_torch.repeat_interleave(S)
    x_original_expanded = X_train_torch.unsqueeze(1).expand(-1, S, -1).reshape(-1, D)

    weights = None
    if method == "SVGD":
        particles = X_train_torch.unsqueeze(1).repeat(1, S, 1) + torch.randn(B, S, D, device=device) * np.sqrt(EPSILON / 2)
        hist_grad = torch.zeros_like(particles)
    else:
        particles = x_original_expanded.clone()
        if method == 'WFR':
            weights = torch.ones((B, S), device=device) / S
            wt_lr, weight_exponent = lr * wfr_times, 1 - LAMBDA_PARAM * EPSILON * (lr * wfr_times)

    for step in tqdm(range(inner_steps), desc=f"Generating {method} frames"):
        with torch.no_grad():
            loss_values = criterion(saa_model(particles.view(-1,D)), y_repeated)
            dist_sq = torch.sum((particles.view(-1,D) - x_original_expanded)**2, dim=1)
            objective_per_sample = loss_values - LAMBDA_PARAM * dist_sq
            if method == 'WFR':
                avg_objective = (objective_per_sample.view(B, -1) * weights).sum(dim=1).mean().item()
            else:
                avg_objective = objective_per_sample.mean().item()
            objective_history.append(avg_objective)

        particles.requires_grad_(True)
        loss_val = criterion(saa_model(particles.view(-1,D)), y_repeated).sum()
        grads, = torch.autograd.grad(loss_val, particles)
        particles = particles.detach()

        if method == "SVGD":
            grad_log_px = -2 * LAMBDA_PARAM * (particles.view(-1, D) - x_original_expanded)
            total_grad = (grads.view(-1,D) + grad_log_px).view(B, S, D)/LAMBDA_PARAM/EPSILON
            K, grad_K_x = rbf_kernel_batched_torch(particles)
            svgd_grad = (torch.bmm(K, total_grad) + torch.sum(grad_K_x, dim=2)) / S
            with torch.no_grad():
                hist_grad = adagrad_hist_decay * hist_grad + (1 - adagrad_hist_decay) * (svgd_grad**2)
                particles += lr * svgd_grad / (1e-6 + torch.sqrt(hist_grad))
        elif method == 'MMD':
            _, grad_K_clone = imq_kernel_and_grad(particles, particles)
            _, grad_K_orig = imq_kernel_and_grad(particles, X_train_torch)
            mmd_grad = -LAMBDA_PARAM * (torch.mean(grad_K_clone, dim=1) - torch.mean(grad_K_orig, dim=1))
            velocity = grads.view(-1,D) + mmd_grad
            particles += lr * velocity + torch.randn_like(particles) * MMD_NOISE_STD
        else: 
            anchor_grad = -2 * LAMBDA_PARAM * (particles - x_original_expanded)
            velocity = grads.view(-1,D) + anchor_grad
            if method == 'WRM':
                particles += lr * velocity
            else:
                particles += lr * velocity + torch.randn_like(particles) * torch.sqrt(torch.tensor(2*lr*LAMBDA_PARAM*EPSILON, device=device))
                if method == 'WFR':
                    with torch.no_grad():
                        new_obj = criterion(saa_model(particles), y_repeated).view(B,S) - LAMBDA_PARAM * torch.sum((particles - x_original_expanded)**2, dim=1).view(B,S)
                        weights = (weights ** weight_exponent) * torch.exp(new_obj * wt_lr)
                        weights /= (torch.sum(weights, dim=1, keepdim=True) + 1e-9)
                    
                    with torch.no_grad():
                        low_weight_mask = weights < 1e-4
                        rows_with_low_weights = torch.any(low_weight_mask, dim=1)
                        if torch.any(rows_with_low_weights):
                            x_reshaped = particles.view(B, S, -1)
                            max_weight_vals, max_weight_indices = torch.max(weights, dim=1, keepdim=True)
                            highest_weight_point_data = torch.gather(x_reshaped, 1, max_weight_indices.unsqueeze(-1).expand(-1, -1, D))
                            
                            low_weights_sum = torch.sum(weights * low_weight_mask, dim=1, keepdim=True)
                            num_low_weights = torch.sum(low_weight_mask, dim=1, keepdim=True, dtype=weights.dtype)

                            avg_weight = (max_weight_vals + low_weights_sum) / (num_low_weights + 1.0 + 1e-9)
                            avg_weight_expanded = avg_weight.expand_as(weights)

                            max_weight_mask = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, max_weight_indices, True)
                            update_mask = (low_weight_mask | max_weight_mask) & rows_with_low_weights.unsqueeze(1)
                            weights = torch.where(update_mask, avg_weight_expanded, weights)

                            x_update_mask = low_weight_mask & rows_with_low_weights.unsqueeze(1)
                            replacement_data = highest_weight_point_data.expand_as(x_reshaped)
                            x_reshaped = torch.where(x_update_mask.unsqueeze(-1), replacement_data, x_reshaped)
                            
                            particles = x_reshaped.view(-1, D)
                            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)
        
        if step % 10 == 0:
            frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
            plot_frame_samples(saa_model, X_train, y_train, particles.view(-1,D), f"Perturbation Process ({method})",
                               frame_path, "SAA", f"Step: {step+1}", weights)
            gif_filenames.append(frame_path)

    if imageio and gif_filenames:
        gif_path = os.path.join(output_dir, f"saa_perturbation_{method}_no_forces.gif")
        imageio.mimsave(gif_path, [imageio.imread(f) for f in gif_filenames], duration=0.1)
        print(f"GIF saved to {gif_path}")
        
    return objective_history

def plot_objective_comparison(histories: Dict[str, list], save_path_png: str):
    plt.figure(figsize=(3.8, 2.613), dpi=300, tight_layout=True)
    for name, history in histories.items():
        if history: 
            plt.plot(history, label=name, linewidth=1.5)
    plt.xlabel("iteration")
    plt.ylabel("loss") 
    # plt.title("Perturbation Objective Comparison")
    plt.legend()
    plt.grid(False)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    save_path_pdf = save_path_png.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_png); plt.savefig(save_path_pdf); plt.close()
    print(f"Objective comparison plot saved to {save_path_png} and {save_path_pdf}")


# --- 5. Main Execution ---
if __name__ == '__main__':
    plt.style.use('jz.mplstyle')
    X_train, y_train = create_dataset(n_samples=N_SAMPLES_TRAIN, noise=0.1, imbalance_ratio=0.8, random_state=SEED)
    X_test, y_test = create_dataset(n_samples=N_SAMPLES_TEST, noise=0.3, random_state=SEED+1)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_test_torch = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_torch = torch.from_numpy(y_test).long().to(DEVICE)


    LOG_EPOCH_INTERVAL = 10
    ROBUSTNESS_LOG_INTERVAL = 10

    model_names = ["SAA", "SVGD", "MMD", "Dual", "WGF", "WRM", "WFR"]
    loss_histories = {name: [] for name in ["SVGD", "Dual", "WGF", "WFR", "MMD"]}

    robustness_history = {name: {"FGSM": [], "PGD": []} for name in model_names}

    criterion = nn.CrossEntropyLoss()

    # --- SAA Training ---
    print("\n--- Training Standard (SAA) Model ---")
    saa_model = Classifier(INPUT_DIM, hidden_dim=32).to(DEVICE)
    optimizer_saa = optim.Adam(saa_model.parameters(), lr=LR)

    for epoch in tqdm(range(MAX_EPOCHS), desc="SAA Training"):
        epoch_loss = 0
        saa_model.train()
        for X_batch, y_batch in train_loader:
            optimizer_saa.zero_grad()
            loss = criterion(saa_model(X_batch.to(DEVICE)), y_batch.to(DEVICE))
            loss.backward()
            optimizer_saa.step()
            epoch_loss += loss.item()

    trained_models = {"SAA": saa_model}

    all_methods = ["WRM", "SVGD", "WGF", "WFR"] #, , "MMD"
    comparison_histories = {}
    for method in all_methods:
        history = plot_perturbation_process(saa_model, X_train, y_train, method, OUTPUT_DIR, lr=0.01, inner_steps=300)
        if method in ["WGF", "WFR", "WRM", "SVGD"]:
            comparison_histories[method] = history

    plot_objective_comparison(comparison_histories, os.path.join(OUTPUT_DIR, "saa_perturb_objective_comparison.png"))

    #Generate WITH-FORCE GIFs
    for method in all_methods:
        plot_perturbation_process_kernel_force(saa_model, X_train, y_train, method, OUTPUT_DIR, lr=inner_lr)


    # --- DRO Models Training ---
    dro_methods_to_train = ["SVGD", "WRM", "Dual"] #"MMD", "WFR", "WGF", 

    for method in dro_methods_to_train:
        print(f"\n--- Training {method} Model ---")
        model = Classifier(INPUT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        method_frames_dir = os.path.join(OUTPUT_DIR, f"{method}_frames")
        os.makedirs(method_frames_dir, exist_ok=True)
        method_frames_dir_pdf = os.path.join(OUTPUT_DIR_PDF, f"{method}_frames")
        os.makedirs(method_frames_dir_pdf, exist_ok=True)

        gif_filenames_dis = []
        gif_filenames_sam = []

        for epoch in tqdm(range(MAX_EPOCHS), desc=f"{method} Training"):
            epoch_loss = 0
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                if method == "Dual":
                    levels = np.arange(SINKHORN_SAMPLE_LEVEL + 1); numerators = 2.0**(-levels); denominator = 2.0 - 2.0**(-SINKHORN_SAMPLE_LEVEL)
                    probabilities = numerators / denominator; sampled_level = np.random.choice(levels, p=probabilities)
                    m = 2 ** sampled_level
                    X_perturbed = sinkhorn_base_sampler(X_batch, m_samples=m)
                    y_repeated = y_batch.repeat_interleave(m); predictions = model(X_perturbed)
                    loss = compute_sinkhorn_loss(predictions, y_repeated, m, LAMBDA_PARAM * EPSILON)

                elif method == "WGF":
                    X_perturbed = wgf_sampler(X_batch, y_batch, model, epoch)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    model.train()
                    loss = criterion(model(X_perturbed), y_repeated)

                elif method == "WRM":
                    model.eval()
                    X_perturbed = wrm_sampler(X_batch, y_batch, model, epoch)
                    y_repeated = y_batch
                    record_loss = criterion(model(X_perturbed), y_repeated)
                    loss = record_loss

                elif method == "WFR":
                    model.eval()
                    X_perturbed, weights = wfr_sampler(X_batch, y_batch, model, epoch)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    model.train()
                    predictions = model(X_perturbed)
                    loss_values = nn.CrossEntropyLoss(reduction='none')(predictions, y_repeated)
                    weighted_loss = (loss_values.view(-1, NUM_SAMPLES_PER_POINT) * weights).sum(dim=1).mean()
                    loss = weighted_loss
                elif method == "MMD":
                    model.eval()
                    X_perturbed = mmd_dro_sampler(X_batch, y_batch, model, epoch)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    model.train()
                    loss = criterion(model(X_perturbed), y_repeated)
                elif method == "SVGD":
                    model.eval()
                    X_perturbed = svgd_sampler(X_batch, y_batch, model)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    model.train()
                    loss = criterion(model(X_perturbed), y_repeated)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item()

            if epoch % LOG_EPOCH_INTERVAL == 0 :
                if method != "WRM":
                    loss_histories[method].append(epoch_loss / len(train_loader))

            if method in ["SVGD", "WGF", "WRM", "WFR", "MMD"]:
                model.eval()
                epoch_weights = None 
                if method == "WGF":
                    epoch_X_p = wgf_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "WRM":
                    epoch_X_p = wrm_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "WFR":
                    epoch_X_p, epoch_weights = wfr_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "MMD":
                    epoch_X_p = mmd_dro_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "SVGD":
                    epoch_X_p = svgd_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)
                frame_path = os.path.join(method_frames_dir, f"{method}_frame_{epoch:03d}.png")
                plot_frame(model, X_train, y_train, epoch_X_p, f"Worst-case Distribution({method})", frame_path, method, epoch)
                
                samples_frame_path = os.path.join(method_frames_dir, f"{method}_frame_{epoch:03d}_samples.png")
                plot_frame_samples(model, X_train, y_train, epoch_X_p, f"Worst-case Samples({method})", samples_frame_path, method, epoch, weights=epoch_weights)
                
                gif_filenames_dis.append(frame_path)
                gif_filenames_sam.append(samples_frame_path)

        trained_models[method] = model

        if imageio and method in ["WGF", "WRM", "WFR", "MMD", "SVGD"]:
            print(f"Creating {method} evolution GIF...")
            gif_path = f"{OUTPUT_DIR}/{method}_two_moons_evolution.gif"
            with imageio.get_writer(gif_path, mode='I', duration=200, loop=0, format='gif') as writer:
                for filename in gif_filenames_dis:
                    writer.append_data(imageio.imread(filename))
            with imageio.get_writer(gif_path.replace(".gif", "_samples.gif"), mode='I', duration=200, loop=0, format='gif') as writer:
                for filename in gif_filenames_sam:
                    writer.append_data(imageio.imread(filename))
            print(f"GIF saved to {gif_path}")

    print("\n--- Generating Final Plots and Evaluations ---")

    # 1. Loss Comparison Plot
    plot_loss_comparison(loss_histories, f"{OUTPUT_DIR}/loss_comparison.png")
    print(f"Loss comparison plot saved to {OUTPUT_DIR}/loss_comparison.png and {OUTPUT_DIR_PDF}/loss_comparison.pdf")

    # 2. Boundary Comparison Plot
    visualize_all_boundaries(trained_models, X_train, y_train, "Decision Boundaries", f"{OUTPUT_DIR}/final_boundary_comparison.png")
    print(f"Final boundary comparison plot saved to {OUTPUT_DIR}/final_boundary_comparison.png and {OUTPUT_DIR_PDF}/final_boundary_comparison.pdf")

    # 4. Final Robustness Evaluation
    # MODIFIED: Final evaluation with adversarial attacks
    # print("\n--- Evaluating Final Model Accuracy---")
    # print("-" * 65)
    # for name, model in trained_models.items():
    #     model.eval()
    #     acc_clean = evaluate_accuracy(model, X_test_torch, y_test_torch)
    #     print(f"| Model: {name:<12} | Clean Acc: {acc_clean:7.2%} |")
    # print("-" * 65)

    print(f"\nExperiment complete. All results saved in '{OUTPUT_DIR}' and '{OUTPUT_DIR_PDF}' directories.")