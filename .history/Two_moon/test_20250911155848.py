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
# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb # Added for RGBA color creation

# Attempt to import imageio for GIF creation
try:
    import imageio
except ImportError:
    print("The 'imageio' library is not installed. GIF generation will be skipped.")
    print("Please install it to create GIFs: pip install imageio")
    imageio = None

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Experiment parameters
N_SAMPLES_TRAIN = 100
N_SAMPLES_TEST = 1000
INPUT_DIM = 2
HIDDEN_DIM = 32
OUTPUT_DIM = 2
MAX_EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-2
SEED = 100

# DRO parameters
EPSILON = 0.01
LAMBDA_PARAM = 0.5
NUM_SAMPLES_PER_POINT = 5
SINKHORN_SAMPLE_LEVEL = 5
inner_lr = 1e-2
inner_steps = 50

# Attack parameters
ATTACK_EPSILON = 0.3 # Strength of the adversarial attack
PGD_ALPHA = 0.05     # Step size for PGD
PGD_ITERS = 5       # Number of iterations for PGD

# Setup output directory
OUTPUT_DIR = "two_moons_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 2. Model and Data ---
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=64):
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

def get_true_boundary_model(input_dim):
    print("--- Training True Boundary Model ---")
    from sklearn.datasets import make_moons
    X_true, y_true = make_moons(n_samples=4000, noise=0.15, random_state=42)
    true_model = Classifier(input_dim).to(DEVICE)
    optimizer = optim.Adam(true_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_true).float().to(DEVICE),
                                      torch.from_numpy(y_true).long().to(DEVICE)),
                        batch_size=128, shuffle=True)
    true_model.train()
    for _ in range(50):
        for X_batch, y_batch in loader:
            optimizer.zero_grad(); loss = criterion(true_model(X_batch), y_batch); loss.backward(); optimizer.step()
    true_model.eval()
    return true_model

# --- 3. DRO Samplers and Loss Functions ---
def rgo_sampler(x_original_batch, y_original_batch, model, epoch, inner_lr=inner_lr, inner_steps=inner_steps):
    batch_size = x_original_batch.shape[0]
    x_pert_batch = x_original_batch.clone().detach()
    steps = int(min(5, inner_steps * (epoch + 1) / MAX_EPOCHS))
    for _ in range(steps):
        x_pert_batch.requires_grad_(True)
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_pert_batch), y_original_batch)
        grads, = torch.autograd.grad(loss_values, x_pert_batch, grad_outputs=torch.ones_like(loss_values))
        x_pert_batch = x_pert_batch.detach()
        grad_total = -grads / LAMBDA_PARAM + 2 * (x_pert_batch - x_original_batch)
        x_pert_batch -= inner_lr * grad_total

    x_opt_star_batch = x_pert_batch
    var_rgo = EPSILON
    if var_rgo <= 1e-12:
        return x_opt_star_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    std_rgo = math.sqrt(var_rgo)
    f_model_loss_opt_star = nn.CrossEntropyLoss(reduction='none')(model(x_opt_star_batch), y_original_batch)
    norm_sq_opt_star = torch.sum((x_opt_star_batch - x_original_batch.detach()) ** 2, dim=1)
    f_L_xi_opt_star = (-f_model_loss_opt_star / (LAMBDA_PARAM * EPSILON)) + (norm_sq_opt_star / EPSILON)
    x_opt_star_3d = x_opt_star_batch.unsqueeze(1)
    x_original_3d = x_original_batch.detach().unsqueeze(1)
    f_L_xi_opt_star_3d = f_L_xi_opt_star.unsqueeze(1)
    final_accepted_perturbations = torch.zeros((batch_size, NUM_SAMPLES_PER_POINT, INPUT_DIM), device=DEVICE)
    active_flags = torch.ones((batch_size, NUM_SAMPLES_PER_POINT), dtype=torch.bool, device=DEVICE)
    for _ in range(40):
        if not active_flags.any(): break
        pert_proposals = torch.randn_like(final_accepted_perturbations) * std_rgo
        x_candidates = x_opt_star_3d + pert_proposals
        x_candidates_flat = x_candidates.view(-1, INPUT_DIM)
        y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
        f_model_loss_candidates = nn.CrossEntropyLoss(reduction='none')(model(x_candidates_flat), y_repeated).view(batch_size, NUM_SAMPLES_PER_POINT)
        norm_sq_candidates = torch.sum((x_candidates - x_original_3d) ** 2, dim=2)
        f_L_xi_candidates = (-f_model_loss_candidates / (LAMBDA_PARAM * EPSILON)) + (norm_sq_candidates / EPSILON)
        diff_cand_opt_norm_sq = torch.sum(pert_proposals**2, dim=2)
        exponent_term3 = diff_cand_opt_norm_sq / (2 * var_rgo)
        acceptance_probs = torch.exp(torch.clamp(-f_L_xi_candidates + f_L_xi_opt_star_3d + exponent_term3, max=10))
        newly_accepted_mask = (torch.rand_like(acceptance_probs) < acceptance_probs) & active_flags
        final_accepted_perturbations[newly_accepted_mask] = pert_proposals[newly_accepted_mask]
        active_flags[newly_accepted_mask] = False
    return (x_opt_star_3d + final_accepted_perturbations).view(-1, INPUT_DIM)

def ld_sampler(x_original_batch, y_original_batch, model):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_original_batch)
    grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
    mean = x_original_batch + grads / (2 * LAMBDA_PARAM)
    std_dev = torch.sqrt(torch.tensor(EPSILON, device=DEVICE))
    mean_expanded = mean.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1)
    noise = torch.randn_like(mean_expanded) * std_dev
    return (mean_expanded + noise).view(-1, INPUT_DIM)
 
def mul_ld_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    x_clone = x_clone.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).reshape(-1, INPUT_DIM)
    for _ in range(int(min(5, inner_steps * (epoch + 1) / MAX_EPOCHS))):
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

    wt_lr = lr * 16
    weight_exponent = 1 - LAMBDA_PARAM * EPSILON * wt_lr

    for _ in range(int(min(5, inner_steps * (epoch + 1) / MAX_EPOCHS))):
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
            new_loss = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated).view(batch_size, -1)

            weights = weights ** weight_exponent * torch.exp(-new_loss * wt_lr)

            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

    return x_clone.detach(), weights.detach()
def sinha_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    for _ in range(int(min(5, inner_steps * (epoch + 1) / MAX_EPOCHS))):
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_original_batch)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone +  lr * (grads  - 2*LAMBDA_PARAM*(x_clone - x_original_batch))
    return x_clone.detach()

def rbf_kernel_full_numpy(X):
    N, D = X.shape
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)

    all_dist_sq = dist_sq.flatten()
    if all_dist_sq.size > 1:
        h2 = 0.5 * np.median(all_dist_sq) / np.log(N + 1.0)
    else:
        h2 = 1.0
    h2 = np.maximum(h2, 1e-6)

    K = np.exp(-dist_sq / (2 * h2))
    grad_K_x = -diff / h2 * K[..., np.newaxis]
    return K, grad_K_x


def svgd_sampler_independent(
    x_original_batch,
    y_original_batch,
    model,
    num_samples_per_point=10,
    lr=1e-1,
    inner_steps=20,
    lambda_param=1.0,
    adagrad_hist_decay=0.9
):

    device = x_original_batch.device

    final_particle_sets = []

    for i in range(len(x_original_batch)):
        x_point = x_original_batch[i:i+1]
        y_label = y_original_batch[i:i+1]

        x_orig_repeated = x_point.repeat(num_samples_per_point, 1)
        y_repeated = y_label.repeat(num_samples_per_point)

        particles = x_orig_repeated.clone().detach().cpu().numpy()
        particles = particles + np.random.randn(*particles.shape) * np.sqrt(EPSILON/2)
        hist_grad = np.zeros_like(particles)

        for _ in range(inner_steps):
            x_tensor = torch.from_numpy(particles).float().to(device).requires_grad_(True)

            neg_log_likelihood = -nn.CrossEntropyLoss(reduction='sum')(model(x_tensor), y_repeated)
            grad_log_py_x, = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_tensor)
            grad_log_px = -2 * lambda_param * (x_tensor - x_orig_repeated)
            total_grad = (grad_log_py_x + grad_log_px).detach().cpu().numpy()

            N = particles.shape[0]
            K, grad_K_x = rbf_kernel_full_numpy(particles)
            svgd_grad = (K @ total_grad + np.sum(grad_K_x, axis=1)) / N

            hist_grad = adagrad_hist_decay * hist_grad + (1 - adagrad_hist_decay) * (svgd_grad**2)
            adj_grad = svgd_grad / (1e-6 + np.sqrt(hist_grad))

            particles = particles + lr * adj_grad

        final_particle_sets.append(particles)

    all_final_particles_np = np.concatenate(final_particle_sets, axis=0)

    return torch.from_numpy(all_final_particles_np).float().to(device)

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
def plot_frame(model, true_boundary_model, X, y, X_perturbed, title, save_path, method, epoch):
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
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7); ax.set_title(title, fontsize=9); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    plt.savefig(save_path); plt.close(fig)

def plot_frame_samples(model, true_boundary_model, X, y, X_perturbed, title, save_path, method, epoch, weights=None):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)

    # 1. Plot original data points
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5, s=20, linewidths=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5, s=20, linewidths=0.5)

    # 2. If perturbed data exists, plot it with connections and variable transparency
    if X_perturbed is not None:
        X_p_np = X_perturbed.cpu().numpy()
        num_repeats = X_perturbed.shape[0] // X.shape[0]
        y_repeated = np.repeat(y, num_repeats)

        # 2a. Handle weights: create uniform weights if not provided
        if weights is None:
            weights = torch.ones((X.shape[0], num_repeats), device=X_perturbed.device) / num_repeats

        weights_flat = weights.flatten().cpu().numpy()

        # 2b. Scale weights to an alpha range
        min_alpha, max_alpha = 0, 0.5
        w_min, w_max = weights_flat.min(), weights_flat.max()
        if w_max > w_min:
            alphas = min_alpha + (weights_flat - w_min) / (w_max - w_min) * (max_alpha - min_alpha)
        else: # Handle uniform weights case
            alphas = np.full_like(weights_flat, max_alpha)

        # 2c. Create a per-point RGBA color array
        colors_rgba = np.zeros((X_p_np.shape[0], 4))
        color_pos_rgb = to_rgb('darkorange')
        color_neg_rgb = to_rgb('dodgerblue')

        pos_mask = y_repeated == 1
        neg_mask = y_repeated == 0

        colors_rgba[pos_mask, :3] = color_pos_rgb
        colors_rgba[neg_mask, :3] = color_neg_rgb
        colors_rgba[:, 3] = alphas # Set alpha channel based on weights

        # 2d. Plot all perturbed points using the RGBA color array
        ax.scatter(X_p_np[:, 0], X_p_np[:, 1], c=colors_rgba, marker='o',edgecolors="gray", linewidths=0.2, s=15)

        # 2e. Connect each original point to its corresponding perturbed points
        for i in range(X.shape[0]):
            original_point = X[i]
            start_index = i * num_repeats
            end_index = start_index + num_repeats
            perturbed_points_block = X_p_np[start_index:end_index, :]

            for p_point in perturbed_points_block:
                ax.plot([original_point[0], p_point[0]],
                        [original_point[1], p_point[1]],
                        color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # 3. Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        Z_model = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)

    ax.contour(xx, yy, Z_model, levels=[0.5], linewidths=1.5, colors='black', linestyles='--')

    # 4. Create legend
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', lw=2, label=f'Boundary ({method})'),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Perturbed Positive', markersize=5, markeredgecolor="gray", markeredgewidth=0.2),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Perturbed Negative', markersize=5, markeredgecolor="gray", markeredgewidth=0.2)
    ]

    # 5. Set plot attributes and save
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.text(0.95, 0.05, f'Epoch: {epoch}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    plt.savefig(save_path)
    plt.close(fig)

def plot_wfr_weights_evolution(model, X_train, y_train, output_dir, lr=1e-2, inner_steps=50):
    """
    Analyzes and plots the evolution of weights in the WFR sampler for a specific point.
    The point chosen is the negative sample furthest to the bottom-right.
    """
    print("--- Analyzing WFR weight evolution for a specific sample ---")
    
    # --- 1. Select the target point ---
    # Find negative samples
    neg_samples_mask = (y_train == 0)
    neg_samples = X_train[neg_samples_mask]
    
    # Heuristic for "bottom-right": maximize x, minimize y -> maximize x - y
    if len(neg_samples) == 0:
        print("Warning: No negative samples found to analyze. Skipping weight evolution plot.")
        return
        
    vals = neg_samples[:, 0] - neg_samples[:, 1]
    target_idx_in_neg = np.argmax(vals)
    target_point_np = neg_samples[target_idx_in_neg]

    # Convert to tensor for the model
    target_point = torch.from_numpy(target_point_np).float().unsqueeze(0).to(DEVICE)
    target_label = torch.tensor([0], dtype=torch.long, device=DEVICE) # We know it's a negative sample

    # --- 2. Run WFR inner loop and record weights ---
    device = target_point.device
    weights = torch.ones((1, NUM_SAMPLES_PER_POINT), dtype=torch.float32, device=device) / NUM_SAMPLES_PER_POINT
    x_clone = target_point.clone().detach().unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = target_label.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = target_point.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).reshape(-1, INPUT_DIM)

    wt_lr = lr * 16
    weight_exponent = 1 - LAMBDA_PARAM * EPSILON * wt_lr
    
    weights_history = []

    model.eval() # Ensure model is in eval mode
    for _ in range(inner_steps):
        # Store a copy of the current weights for this iteration
        weights_history.append(weights.clone().squeeze(0).cpu().numpy())
        
        x_clone.requires_grad_(True)
        
        # Calculate gradients w.r.t. input
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone.detach()

        # Update particle positions (Langevin dynamics step)
        mean = x_clone + lr * (grads - 2 * LAMBDA_PARAM * (x_clone - x_original_expanded))
        std_dev = torch.sqrt(torch.tensor(2 * lr * LAMBDA_PARAM * EPSILON, device=device))
        noise = torch.randn_like(mean) * std_dev
        x_clone = mean + noise

        # Update weights based on the new positions
        with torch.no_grad():
            new_loss = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated).view(1, -1)
            weights = weights ** weight_exponent * torch.exp(-new_loss * wt_lr)
            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

    # Convert history to a plottable numpy array
    weights_history = np.array(weights_history) # Shape: (inner_steps, NUM_SAMPLES_PER_POINT)
    
    # --- 3. Plot the results ---
    plt.style.use('jz.mplstyle')
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)
    
    iterations = range(inner_steps)
    colors = plt.cm.viridis(np.linspace(0, 1, NUM_SAMPLES_PER_POINT))

    for i in range(NUM_SAMPLES_PER_POINT):
        ax.plot(iterations, weights_history[:, i], color=colors[i], label=f'Sample {i+1}')

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weight")
    ax.set_title("WFR Weight Evolution for a Single Point")
    # Place legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.grid(False)
    ax.set_ylim(bottom=0) # Weights can't be negative

    save_path = os.path.join(output_dir, "wfr_weights_evolution.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"WFR weight evolution plot saved to {save_path}")

def visualize_all_boundaries(models: Dict[str, nn.Module], true_boundary_model, X, y, title, save_path):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5, linewidths=0.5, s=20)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5, linewidths=0.5, s=20)
    x_min, x_max, y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1, X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    linestyles = ['--', ':', '-.', '--']
    # true_boundary_model.eval()
    # with torch.no_grad():
    #     Z = true_boundary_model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    # ax.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles='-', linewidths=2)
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=linestyles[i % len(linestyles)], linewidths=1.5)

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
    legend_elements.extend([Line2D([0], [0], color=colors[i], lw=1.5, linestyle=linestyles[i % len(linestyles)], label=name) for i, name in enumerate(models.keys())])
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1)); ax.set_title(title)
    ax.set_xlabel('feature 1'); ax.set_ylabel('feature 2'); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    plt.savefig(save_path); plt.close(fig)

def plot_loss_comparison(loss_histories: Dict[str, list], save_path):
    plt.figure()
    for name, history in loss_histories.items():
        epochs_logged = range(1, (len(history) * LOG_EPOCH_INTERVAL) + 1, LOG_EPOCH_INTERVAL)
        plt.plot(epochs_logged, history, label=name, marker='o',  markersize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_robustness_comparison(history: Dict[str, Dict[str, list]], save_dir: str):
    attack_types = ["FGSM", "PGD"] # MODIFIED
    for attack in attack_types:
        plt.figure(figsize=(4.2, 2.613), dpi=300, tight_layout=True)

        for model_name, attack_history in history.items():
            accuracies = attack_history[attack]
            if not accuracies:
                continue
            epochs_logged = range(1, (len(accuracies) * ROBUSTNESS_LOG_INTERVAL) + 1, ROBUSTNESS_LOG_INTERVAL)
            plt.plot(epochs_logged, accuracies, label=model_name, marker='o', markersize=8)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Testing accuracy under {attack} Attack")
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.grid(False)
        plt.ylim(0, 1.05)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        save_path = os.path.join(save_dir, f"robustness_comparison_{attack.lower()}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"{attack} robustness plot saved to {save_path}")

# --- 5. Attack and Evaluation Functions ---
# MODIFIED SECTION START
def fgsm_attack(model, loss_fn, data, labels, epsilon):
    """Performs the FGSM adversarial attack."""
    data_adv = data.clone().detach().requires_grad_(True)
    outputs = model(data_adv)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Collect the gradients
    grad = data_adv.grad.data
    # Create the perturbed data
    data_adv = data_adv.detach() + epsilon * grad.sign()
    return data_adv

def pgd_attack(model, loss_fn, data, labels, epsilon, alpha, iters):
    """Performs the PGD adversarial attack."""
    data_adv = data.clone().detach()
    # Start with a random perturbation
    data_adv = data_adv + torch.empty_like(data_adv).uniform_(-epsilon, epsilon)

    for _ in range(iters):
        data_adv.requires_grad_(True)
        outputs = model(data_adv)
        loss = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()

        grad = data_adv.grad.data
        data_adv = data_adv.detach() + alpha * grad.sign()

        # Project perturbation back to the epsilon-ball
        eta = torch.clamp(data_adv - data, min=-epsilon, max=epsilon)
        data_adv = data.clone().detach() + eta

    return data_adv
# MODIFIED SECTION END

def evaluate_accuracy(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        predicted = outputs.argmax(dim=1)
        correct = (predicted == y_test).sum().item()
    return correct / len(y_test)

# --- 6. Main Execution ---
if __name__ == '__main__':
    # --- Data ---
    plt.style.use('jz.mplstyle')
    X_train, y_train = create_dataset(n_samples=N_SAMPLES_TRAIN, noise=0.1, imbalance_ratio=0.7, random_state=SEED)
    X_test, y_test = create_dataset(n_samples=N_SAMPLES_TEST, noise=0.3, random_state=SEED+1)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_test_torch = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_torch = torch.from_numpy(y_test).long().to(DEVICE)

    # --- Models & History Logging---
    true_boundary_model = get_true_boundary_model(INPUT_DIM)

    LOG_EPOCH_INTERVAL = 10
    ROBUSTNESS_LOG_INTERVAL = 10

    model_names = ["SAA", "RGO", "Dual", "MultiLD", "WDRO", "WFR"]#"LD", "SVGD",
    loss_histories = {name: [] for name in ["RGO", "Dual", "MultiLD", "WFR"]}  # SAA does not use DRO loss "LD", "WDRO", "SVGD",
    # MODIFIED: Changed attack names
    robustness_history = {name: {"FGSM": [], "PGD": []} for name in model_names}

    criterion = nn.CrossEntropyLoss()

    # --- SAA Training ---
    print("\n--- Training Standard (SAA) Model ---")
    saa_model = Classifier(INPUT_DIM).to(DEVICE)
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

        # MODIFIED: Evaluate with adversarial attacks
        if (epoch + 1) % ROBUSTNESS_LOG_INTERVAL == 0:
            saa_model.eval()
            X_test_fgsm = fgsm_attack(saa_model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON)
            X_test_pgd = pgd_attack(saa_model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON, PGD_ALPHA, PGD_ITERS)

            acc_fgsm = evaluate_accuracy(saa_model, X_test_fgsm, y_test_torch)
            acc_pgd = evaluate_accuracy(saa_model, X_test_pgd, y_test_torch)
            robustness_history["SAA"]["FGSM"].append(acc_fgsm)
            robustness_history["SAA"]["PGD"].append(acc_pgd)

    trained_models = {"SAA": saa_model}

    # --- DRO Models Training ---
    dro_methods_to_train = ["WFR", "WDRO", "MultiLD", "RGO", "Dual"] #"LD",  , "SVGD"

    for method in dro_methods_to_train:
        print(f"\n--- Training {method} Model ---")
        model = Classifier(INPUT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
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

                elif method in ["RGO", "LD", "MultiLD"]:
                    model.eval()
                    if method == "MultiLD":
                        X_perturbed = mul_ld_sampler(X_batch, y_batch, model, epoch)
                    elif method == "RGO":
                        X_perturbed = rgo_sampler(X_batch, y_batch, model, epoch)
                    else:   # LD
                        X_perturbed = ld_sampler(X_batch, y_batch, model)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    model.train()
                    #record_loss = compute_sinkhorn_loss(model(sinkhorn_base_sampler(X_batch, m_samples=NUM_SAMPLES_PER_POINT)), y_repeated, NUM_SAMPLES_PER_POINT, LAMBDA_PARAM * EPSILON)
                    loss = criterion(model(X_perturbed), y_repeated)

                elif method == "WDRO":
                    model.eval()
                    X_perturbed = sinha_sampler(X_batch, y_batch, model, epoch)
                    y_repeated = y_batch
                    record_loss = criterion(model(X_perturbed), y_repeated)
                    loss = record_loss

                elif method == "SVGD":
                    model.eval()
                    X_perturbed = svgd_sampler_independent(X_batch, y_batch, model, num_samples_per_point=NUM_SAMPLES_PER_POINT, lr=1e-2, inner_steps=50, lambda_param=LAMBDA_PARAM)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    model.train()
                    #record_loss = compute_sinkhorn_loss(model(sinkhorn_base_sampler(X_batch, m_samples=NUM_SAMPLES_PER_POINT)), y_repeated, NUM_SAMPLES_PER_POINT, LAMBDA_PARAM * EPSILON)
                    loss = criterion(model(X_perturbed), y_repeated)
                elif method == "WFR":
                    model.eval()
                    X_perturbed, weights = wfr_sampler(X_batch, y_batch, model, epoch)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    model.train()
                    predictions = model(X_perturbed)
                    loss_values = nn.CrossEntropyLoss(reduction='none')(predictions, y_repeated)
                    weighted_loss = (loss_values.view(-1, NUM_SAMPLES_PER_POINT) * weights).sum(dim=1).mean()
                    #record_loss = compute_sinkhorn_loss(model(sinkhorn_base_sampler(X_batch, m_samples=NUM_SAMPLES_PER_POINT)), y_repeated, NUM_SAMPLES_PER_POINT, LAMBDA_PARAM * EPSILON)
                    loss = weighted_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item()

            if epoch % LOG_EPOCH_INTERVAL == 0 :
                if method != "WDRO":
                    loss_histories[method].append(epoch_loss / len(train_loader))

            # MODIFIED: Evaluate with adversarial attacks
            if (epoch + 1) % ROBUSTNESS_LOG_INTERVAL == 0:
                model.eval()
                X_test_fgsm = fgsm_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON)
                X_test_pgd = pgd_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON, PGD_ALPHA, PGD_ITERS)

                acc_fgsm = evaluate_accuracy(model, X_test_fgsm, y_test_torch)
                acc_pgd = evaluate_accuracy(model, X_test_pgd, y_test_torch)
                robustness_history[method]["FGSM"].append(acc_fgsm)
                robustness_history[method]["PGD"].append(acc_pgd)

            if method in ["RGO", "LD", "MultiLD", "WDRO", "WFR"]: #"SVGD",
                model.eval()
                epoch_weights = None # Default to None for methods without weights
                if method == "MultiLD":
                    epoch_X_p = mul_ld_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "WDRO":
                    epoch_X_p = sinha_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "SVGD":
                    epoch_X_p = svgd_sampler_independent(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, num_samples_per_point=NUM_SAMPLES_PER_POINT, lr=1e-2, inner_steps=50, lambda_param=LAMBDA_PARAM)
                elif method == "WFR":
                    # Capture weights from the sampler for visualization
                    epoch_X_p, epoch_weights = wfr_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                else:
                    epoch_X_p = rgo_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch) if method == "RGO" else ld_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)

                frame_path = f"{OUTPUT_DIR}/{method}_frame_{epoch:03d}.png"
                plot_frame(model, true_boundary_model, X_train, y_train, epoch_X_p, f"Worst-case Distribution({method})", frame_path, method, epoch)
                # Pass the captured weights to the plotting function
                plot_frame_samples(model, true_boundary_model, X_train, y_train, epoch_X_p, f"Worst-case Samples({method})", frame_path.replace(".png", "_samples.png"), method, epoch, weights=epoch_weights)
                gif_filenames_dis.append(frame_path)
                gif_filenames_sam.append(frame_path.replace(".png", "_samples.png"))

        trained_models[method] = model

        if imageio and method in ["RGO", "MultiLD", "WDRO", "WFR"]: #"LD", "SVGD",
            print(f"Creating {method} evolution GIF...")
            gif_path = f"{OUTPUT_DIR}/{method}_two_moons_evolution.gif"
            with imageio.get_writer(gif_path, mode='I', duration=200, loop=0, format='gif') as writer:
                for filename in gif_filenames_dis:
                    writer.append_data(imageio.imread(filename))
            with imageio.get_writer(gif_path.replace(".gif", "_samples.gif"), mode='I', duration=200, loop=0, format='gif') as writer:
                for filename in gif_filenames_sam:
                    writer.append_data(imageio.imread(filename))
            print(f"GIF saved to {gif_path}")

        if method == "WFR" :
            plot_wfr_weights_evolution(trained_models["WFR"], X_train, y_train, OUTPUT_DIR)

    # --- Final Visualizations and Evaluations ---
    print("\n--- Generating Final Plots and Evaluations ---")

    # Call the new function for WFR weight evolution analysis


    # 1. Loss Comparison Plot
    plot_loss_comparison(loss_histories, f"{OUTPUT_DIR}/loss_comparison.png")
    print(f"Loss comparison plot saved to {OUTPUT_DIR}/loss_comparison.png")

    # 2. Boundary Comparison Plot
    visualize_all_boundaries(trained_models, true_boundary_model, X_train, y_train, "Decision Boundaries", f"{OUTPUT_DIR}/final_boundary_comparison.png")
    print(f"Final boundary comparison plot saved to {OUTPUT_DIR}/final_boundary_comparison.png")

    # 3. Robustness Evolution Plot
    plot_robustness_comparison(robustness_history, OUTPUT_DIR)

    # 4. Final Robustness Evaluation
    # MODIFIED: Final evaluation with adversarial attacks
    print("\n--- Evaluating Final Model Robustness against Adversarial Attacks ---")
    print("-" * 65)
    print(f"Attack Strength (Epsilon): {ATTACK_EPSILON}")
    print("-" * 65)
    for name, model in trained_models.items():
        model.eval()
        X_test_fgsm = fgsm_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON)
        X_test_pgd = pgd_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON, PGD_ALPHA, PGD_ITERS)

        acc_clean = evaluate_accuracy(model, X_test_torch, y_test_torch)
        acc_fgsm = evaluate_accuracy(model, X_test_fgsm, y_test_torch)
        acc_pgd = evaluate_accuracy(model, X_test_pgd, y_test_torch)
        print(f"| Model: {name:<12} | Clean Acc: {acc_clean:7.2%} | FGSM Acc: {acc_fgsm:7.2%} | PGD Acc: {acc_pgd:7.2%} |")
    print("-" * 65)

    print(f"\nExperiment complete. All results saved in '{OUTPUT_DIR}' directory.")
