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
N_SAMPLES_TRAIN = 150
N_SAMPLES_TEST = 1000
INPUT_DIM = 2
HIDDEN_DIM = 32
OUTPUT_DIM = 2
MAX_EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-2
SEED = 100

# DRO parameters
EPSILON = 0.1
LAMBDA_PARAM = 5
NUM_SAMPLES_PER_POINT = 5
SINKHORN_SAMPLE_LEVEL = 4
inner_lr = 1e-2
inner_steps = 200

MMD_NOISE_STD = 1e-3
wfr_times = 16  # Times the inner_lr for WFR sampler

# Attack parameters
ATTACK_EPSILON = 0.3 # Strength of the adversarial attack
PGD_ALPHA = 0.05     # Step size for PGD
PGD_ITERS = 5       # Number of iterations for PGD

# Setup output directory
OUTPUT_DIR = "test"
OUTPUT_DIR_PDF = "test_pdf" # New directory for PDFs
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_PDF, exist_ok=True) # Create the PDF directory
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 2. Model and Data ---
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=16):
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

    # Pairwise squared distances
    X_sq = torch.sum(X**2, dim=1, keepdim=True)
    Y_sq = torch.sum(Y**2, dim=1, keepdim=True)
    XY = torch.matmul(X, Y.T)
    dist_sq = (X_sq - 2*XY + Y_sq.T).clamp(min=0)

    # --- Median Heuristic for bandwidth (based on X's internal distances) ---
    X_dist_sq = (X_sq - 2 * torch.matmul(X, X.T) + X_sq.T).clamp(min=0)
    all_dist_sq_x = X_dist_sq.flatten()

    if all_dist_sq_x.numel() > 1:
        h_sq = 0.5 * torch.median(all_dist_sq_x) / torch.log(torch.tensor(m + 1.0, device=X.device))
    else:
        h_sq = torch.tensor(1.0, device=X.device)
    h_sq = torch.clamp(h_sq, min=1e-6)

    # Kernel matrix
    K = torch.exp(-dist_sq / (2 * h_sq))

    # Kernel gradient: ∇_x k(x, y) = - (x - y) / h^2 * k(x, y)
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
        # Step A: Compute loss gradient ∇f(x_i) for each particle
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
        loss_grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone.detach()

        # Step B: Compute kernel gradients using the provided kernel_fn
        # Term 1: Interaction among perturbed particles (x_clone)
        _, grad_K_clone = kernel_fn(x_clone, x_clone)
        mmd_term1 = torch.mean(grad_K_clone, dim=1) # E_{y~ρ}[∇_x k(x,y)]

        # Term 2: Interaction with original batch points

        _, grad_K_orig = kernel_fn(x_clone, x_original_batch)
        mmd_term2 = torch.mean(grad_K_orig, dim=1) # E_{z~ρ₀}[∇_x k(x,z)]

        # Step C: Compute the total velocity based on the correct MMD-DRO gradient
        velocity = loss_grads - LAMBDA_PARAM * (mmd_term1 - mmd_term2)

        # Step D: Add random noise for Langevin dynamics
        noise = torch.randn_like(x_clone) * MMD_NOISE_STD
        
        # Step E: Update particle positions (gradient ascent + noise)
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

# def rbf_kernel_full_numpy(X):
#     N, D = X.shape
#     diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
#     dist_sq = np.sum(diff**2, axis=-1)

#     all_dist_sq = dist_sq.flatten()
#     if all_dist_sq.size > 1:
#         h2 = 0.5 * np.median(all_dist_sq) / np.log(N + 1.0)
#     else:
#         h2 = 1.0
#     h2 = np.maximum(h2, 1e-6)

#     K = np.exp(-dist_sq / (2 * h2))
#     grad_K_x = -diff / h2 * K[..., np.newaxis]
#     return K, grad_K_x


# def svgd_sampler(
#     x_original_batch,
#     y_original_batch,
#     model,
#     num_samples_per_point=10,
#     lr=1e-1,
#     inner_steps=20,
#     lambda_param=1.0,
#     adagrad_hist_decay=0.9
# ):

#     device = x_original_batch.device

#     final_particle_sets = []

#     for i in range(len(x_original_batch)):
#         x_point = x_original_batch[i:i+1]
#         y_label = y_original_batch[i:i+1]

#         x_orig_repeated = x_point.repeat(num_samples_per_point, 1)
#         y_repeated = y_label.repeat(num_samples_per_point)

#         particles = x_orig_repeated.clone().detach().cpu().numpy()
#         particles = particles + np.random.randn(*particles.shape) * np.sqrt(EPSILON/2)
#         hist_grad = np.zeros_like(particles)

#         for _ in range(inner_steps):
#             x_tensor = torch.from_numpy(particles).float().to(device).requires_grad_(True)

#             neg_log_likelihood = -nn.CrossEntropyLoss(reduction='sum')(model(x_tensor), y_repeated)
#             grad_log_py_x, = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_tensor)
#             grad_log_px = -2 * lambda_param * (x_tensor - x_orig_repeated)
#             total_grad = (grad_log_py_x + grad_log_px).detach().cpu().numpy()

#             N = particles.shape[0]
#             K, grad_K_x = rbf_kernel_full_numpy(particles)
#             svgd_grad = (K @ total_grad + np.sum(grad_K_x, axis=1)) / N

#             hist_grad = adagrad_hist_decay * hist_grad + (1 - adagrad_hist_decay) * (svgd_grad**2)
#             adj_grad = svgd_grad / (1e-6 + np.sqrt(hist_grad))

#             particles = particles + lr * adj_grad

#         final_particle_sets.append(particles)

#     all_final_particles_np = np.concatenate(final_particle_sets, axis=0)

#     return torch.from_numpy(all_final_particles_np).float().to(device)

def rbf_kernel_batched_torch(particles):
    """
    Computes the RBF kernel and its gradient for a batch of particle sets
    using only PyTorch operations.

    Args:
        particles (torch.Tensor): A tensor of shape (B, S, D),
                                  where B is batch size, S is num_samples,
                                  and D is feature dimension.

    Returns:
        tuple: A tuple containing:
            - K (torch.Tensor): The RBF kernel matrix of shape (B, S, S).
            - grad_K_x (torch.Tensor): The gradient of the kernel wrt particles,
                                       shape (B, S, S, D).
    """
    B, S, D = particles.shape
    device = particles.device

    # Calculate pairwise squared distances efficiently using torch.cdist
    # cdist computes (B, S, S), so we square it to get squared distances
    sq_dist = torch.cdist(particles, particles, p=2).pow(2)

    # Use the median heuristic to set the bandwidth h for each item in the batch
    # This makes the bandwidth adaptive to the particle spread for each point
    median_sq_dist = torch.median(sq_dist.view(B, -1), dim=1, keepdim=True)[0]
    # Reshape to (B, 1, 1) for broadcasting
    median_sq_dist = median_sq_dist.view(B, 1, 1)

    # Heuristic for h^2 from the SVGD paper
    # Adding a small epsilon for stability if median is zero
    h_squared =  median_sq_dist / (torch.log(torch.tensor(S, device=device)) + 1e-8)

    # Compute the kernel K
    K = torch.exp(-sq_dist / (2 * h_squared))

    # Compute the gradient of the kernel
    # diff has shape (B, S, S, D)
    diff = particles.unsqueeze(2) - particles.unsqueeze(1)
    # The shape of K needs to be expanded to (B, S, S, 1) for broadcasting
    grad_K_x = -diff / h_squared.unsqueeze(-1) * K.unsqueeze(-1)

    return K, grad_K_x

def svgd_sampler(
    x_original_batch,
    y_original_batch,
    model,
    num_samples_per_point=NUM_SAMPLES_PER_POINT,
    lr=1e-2,
    inner_steps=200,
    lambda_param=LAMBDA_PARAM,
    adagrad_hist_decay=0.9
):

    if x_original_batch.shape[0] == 0:
        return torch.empty(0, x_original_batch.shape[1], device=x_original_batch.device)
    device = x_original_batch.device
    B, D = x_original_batch.shape
    S = num_samples_per_point

    # 1. Prepare Batched Tensors on the correct device
    x_orig_repeated = x_original_batch.unsqueeze(1).repeat(1, S, 1)
    y_repeated = y_original_batch.repeat_interleave(S)

    # 2. Initialize Particles and Adagrad History as Torch Tensors
    particles = x_orig_repeated.clone().detach()
    particles += torch.randn_like(particles) * np.sqrt(EPSILON / 2)
    hist_grad = torch.zeros_like(particles)

    for _ in range(inner_steps):
        # We need to clone and set requires_grad=True in each step
        # to build the computation graph for this iteration
        x_tensor = particles.view(B * S, D).clone().requires_grad_(True)
        
        x_orig_repeated_flat = x_orig_repeated.view(B * S, D)

        # 3. Compute Gradients entirely within PyTorch
        neg_log_likelihood = nn.CrossEntropyLoss(reduction='sum')(model(x_tensor), y_repeated)
        grad_log_py_x, = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_tensor)
        grad_log_px = -2 * lambda_param * (x_tensor - x_orig_repeated_flat)

        total_grad_flat = (grad_log_py_x + grad_log_px)/ (lambda_param * EPSILON)
        total_grad = total_grad_flat.view(B, S, D)

        # 4. Batched Kernel and SVGD Update using Torch
        K, grad_K_x = rbf_kernel_batched_torch(particles)
        
        # Batched matrix multiplication: (B,S,S) @ (B,S,D) -> (B,S,D)
        # torch.bmm is optimized for this
        K_grad_prod = torch.bmm(K, total_grad)
        
        # Summing the gradient of the kernel
        # sum over the 'n' dimension of grad_K_x (B,S,n,D) -> (B,S,D)
        sum_grad_K = torch.sum(grad_K_x, dim=2)

        svgd_grad = (K_grad_prod + sum_grad_K) / S

        # 5. Batched Adagrad Update using Torch Tensors
        # No need to detach svgd_grad as we are done with the graph for this step
        with torch.no_grad():
            hist_grad = adagrad_hist_decay * hist_grad + (1 - adagrad_hist_decay) * (svgd_grad**2)
            adj_grad = svgd_grad / (1e-6 + torch.sqrt(hist_grad))
            particles += lr * adj_grad

    # Reshape final particles from (B, S, D) to (B*S, D)
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
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7); ax.set_title(title, fontsize=9); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    plt.savefig(save_path); plt.close(fig)

def plot_frame_samples(model, X, y, X_perturbed, title, save_path, method, epoch, weights=None):
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
    ax.set_xlabel('feature 1', fontsize=9)
    ax.set_ylabel('feature 2', fontsize=9)
    # Modified to handle both epoch and step
    if isinstance(epoch, int):
        epoch_text = f'Epoch: {epoch}'
    else: # Assuming it's a step counter string
        epoch_text = epoch
    # ax.text(0.95, 0.05, epoch_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    # Save the PNG for GIF creation
    plt.savefig(save_path)
    
    # MODIFICATION: Also save in PDF format
    save_path_pdf = save_path.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_pdf)
    
    plt.close(fig)

def visualize_all_boundaries(models: Dict[str, nn.Module], true_boundary_model, X, y, title, save_path_png):
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
    # true_boundary_model.eval()
    # with torch.no_grad():
    #     Z = true_boundary_model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    # ax.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles='-', linewidths=2)
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
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1)); ax.set_title(title)
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
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    
    save_path_pdf = save_path_png.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    plt.close()

def plot_robustness_comparison(history: Dict[str, Dict[str, list]], save_dir_png: str, save_dir_pdf: str):
    attack_types = ["FGSM", "PGD"] # MODIFIED
    os.makedirs(save_dir_pdf, exist_ok=True) # Ensure PDF directory exists
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

        save_path_png = os.path.join(save_dir_png, f"robustness_comparison_{attack.lower()}.png")
        save_path_pdf = os.path.join(save_dir_pdf, f"robustness_comparison_{attack.lower()}.pdf")
        plt.savefig(save_path_png)
        plt.savefig(save_path_pdf)
        plt.close()
        print(f"{attack} robustness plot saved to {save_path_png} and {save_path_pdf}")

# ==============================================================================
# =========== MODIFIED FUNCTION: `plot_frame_samples_with_forces` ==============
# ==============================================================================
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
        'WGF': (r'Loss grad ($\nabla_z \ell$)', r'Anchor grad ($-2\lambda(z-x)$)'),
        'WFR': (r'Loss grad ($\nabla_z \ell$)', r'Anchor grad ($-2\lambda(z-x)$)'),
        'WRM': (r'Loss grad ($\nabla_z \ell$)', r'Anchor grad ($-2\lambda(z-x)$)'),
        'MMD': (r'Loss grad ($\nabla_z \ell$)', r'Kernel grad ($\nabla_z \text{MMD}$)'),
        'SVGD': (r'Loss Force', r'Repulsive Force')
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
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.text(0.95, 0.05, epoch, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.set_title(title, fontsize=9)
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
        
        if step % 1 == 0:
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

# ==============================================================================
# ================ NEW FUNCTION: `plot_perturbation_process_kernel_force` ======
# ==============================================================================
def plot_perturbation_process_kernel_force(saa_model, X_train, y_train, method, output_dir, lr=0.01, inner_steps=200, adagrad_hist_decay=0.9):
    """
    Visualizes the perturbation process for MMD and SVGD samplers, showing
    the loss-based and kernel-based forces at each step.
    """
    print(f"\n--- Visualizing SAA Perturbation Process for {method} with Forces ---")
    frames_dir = os.path.join(output_dir, f"saa_{method}_forces_frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames_dir_pdf = os.path.join(OUTPUT_DIR_PDF, f"saa_{method}_forces_frames")
    os.makedirs(frames_dir_pdf, exist_ok=True)
    
    gif_filenames = []

    device = next(saa_model.parameters()).device
    saa_model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    X_train_torch = torch.from_numpy(X_train).float().to(device)
    y_train_torch = torch.from_numpy(y_train).long().to(device)
    
    if method == "MMD":
        B, D = X_train_torch.shape
        S = NUM_SAMPLES_PER_POINT
        particles = X_train_torch.unsqueeze(1).expand(-1, S, -1).contiguous().view(-1, D)
        y_repeated = y_train_torch.repeat_interleave(S, dim=0)

        for step in tqdm(range(inner_steps), desc=f"Generating {method} force frames"):
            particles.requires_grad_(True)
            loss_values = criterion(saa_model(particles), y_repeated)
            # Force 1: Gradient from the loss function
            force1_loss_grad, = torch.autograd.grad(loss_values.sum(), particles)
            particles = particles.detach()

            # Force 2: Gradient from the MMD term
            _, grad_K_clone = imq_kernel_and_grad(particles, particles)
            mmd_term1 = torch.mean(grad_K_clone, dim=1) 
            _, grad_K_orig = imq_kernel_and_grad(particles, X_train_torch)
            mmd_term2 = torch.mean(grad_K_orig, dim=1)
            force2_kernel_grad = -LAMBDA_PARAM * (mmd_term1 - mmd_term2)

            # Update particles using the original sampler logic
            velocity = force1_loss_grad + force2_kernel_grad
            noise = torch.randn_like(particles) * MMD_NOISE_STD
            particles = particles + lr * velocity + noise

            # Generate and save a frame for the GIF
            if step % 1 == 0:
                 frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
                 plot_frame_samples_with_forces(
                     model=saa_model, X=X_train, y=y_train, 
                     X_perturbed=particles, 
                     title=f"SAA Perturbation with {method} Forces", 
                     save_path=frame_path, method=method, epoch=f"Step: {step+1}",
                     force1=0.2 * force1_loss_grad,
                     force2=0.2 * force2_kernel_grad
                 )
                 gif_filenames.append(frame_path)

    elif method == "SVGD":
        B, D = X_train_torch.shape
        S = NUM_SAMPLES_PER_POINT
        
        x_orig_repeated = X_train_torch.unsqueeze(1).repeat(1, S, 1)
        y_repeated = y_train_torch.repeat_interleave(S)

        particles = x_orig_repeated.clone().detach()
        particles += torch.randn_like(particles) * np.sqrt(EPSILON / 2)
        hist_grad = torch.zeros_like(particles)

        for step in tqdm(range(inner_steps), desc=f"Generating {method} force frames"):
            x_tensor_flat = particles.view(B * S, D).clone().requires_grad_(True)
            x_orig_repeated_flat = x_orig_repeated.view(B * S, D)

            # Calculate gradients for visualization
            neg_log_likelihood = nn.CrossEntropyLoss(reduction='sum')(saa_model(x_tensor_flat), y_repeated)
            grad_log_py_x, = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_tensor_flat)
            grad_log_px = -2 * LAMBDA_PARAM * (x_tensor_flat - x_orig_repeated_flat)

            # Reshape grads to (B, S, D) for batched kernel multiplication
            grad_log_py_x_shaped = grad_log_py_x.view(B,S,D) / (LAMBDA_PARAM * EPSILON)
            grad_log_px_shaped = grad_log_px.view(B,S,D) / (LAMBDA_PARAM * EPSILON)

            K, grad_K_x = rbf_kernel_batched_torch(particles)
            
            # Force 1: The component of the update driven by the loss
            force1_loss = torch.bmm(K, grad_log_py_x_shaped + grad_log_px_shaped) / S
            
            # Force 2: The component driven by the prior (anchor) and kernel repulsion
            prior_force_term = torch.bmm(K, grad_log_px_shaped)
            repulsive_force_term = torch.sum(grad_K_x, dim=2)
            force2_kernel = (repulsive_force_term) / S #prior_force_term + 

            # Calculate the total update direction, same as in the original sampler
            svgd_grad = force1_loss + force2_kernel
            
            # Update particles using Adagrad
            with torch.no_grad():
                hist_grad = adagrad_hist_decay * hist_grad + (1 - adagrad_hist_decay) * (svgd_grad**2)
                adj_grad = svgd_grad / (1e-6 + torch.sqrt(hist_grad))
                particles += lr * adj_grad

            # Generate and save a frame for the GIF
            if step % 1 == 0:
                 frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
                 plot_frame_samples_with_forces(
                     model=saa_model, X=X_train, y=y_train, 
                     X_perturbed=particles.view(-1, D), 
                     title=f"SAA Perturbation with {method} Forces", 
                     save_path=frame_path, method=method, epoch=f"Step: {step+1}",
                     force1=0.2 * force1_loss.view(-1, D),
                     force2=0.2 * force2_kernel.view(-1, D)
                 )
                 gif_filenames.append(frame_path)

    # Compile the frames into a GIF
    if imageio and gif_filenames:
        print(f"Creating SAA {method} force visualization GIF...")
        gif_path = os.path.join(output_dir, f"saa_perturbation_{method}_forces.gif")
        with imageio.get_writer(gif_path, mode='I', duration=100, loop=0, format='gif') as writer:
            for filename in gif_filenames:
                writer.append_data(imageio.imread(filename))
        print(f"GIF saved to {gif_path}")

def plot_perturbation_process(saa_model, X_train, y_train, method, output_dir, lr=0.01, inner_steps=350, num_samples_per_point=5):
    """
    Visualizes perturbation for WFR, WGF, and WRM.
    Returns the history of the objective values.
    """
    print(f"\n--- Visualizing SAA Perturbation Process for {method} ---")
    frames_dir = os.path.join(output_dir, f"saa_{method}_perturb_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # MODIFICATION: Create parallel directory for PDF frames
    frames_dir_pdf = os.path.join(OUTPUT_DIR_PDF, f"saa_{method}_perturb_frames")
    os.makedirs(frames_dir_pdf, exist_ok=True)
    
    gif_filenames = []
    objective_history = []

    device = next(saa_model.parameters()).device
    saa_model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    X_train_torch = torch.from_numpy(X_train).float().to(device)
    y_train_torch = torch.from_numpy(y_train).long().to(device)
    batch_size = X_train_torch.shape[0]
    
    # For WRM, we want to visualize multiple trajectories, so we adapt it to the multi-particle setting
    num_particles = num_samples_per_point if method != "WRM_single" else 1 # Allow single particle for true WRM viz
    
    x_clone = X_train_torch.clone().detach().unsqueeze(1).expand(-1, num_particles, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = y_train_torch.repeat_interleave(num_particles, dim=0)
    x_original_expanded = X_train_torch.unsqueeze(1).expand(-1, num_particles, -1).reshape(-1, INPUT_DIM)

    weights = None
    if method == 'WFR':
        weights = torch.ones((batch_size, num_particles), dtype=torch.float32, device=device) / num_particles
        wt_lr = lr * wfr_times
        weight_exponent = 1 - LAMBDA_PARAM * EPSILON * wt_lr

    for step in tqdm(range(inner_steps), desc=f"Generating {method} frames"):
        with torch.no_grad():
            loss_values = criterion(saa_model(x_clone), y_repeated)
            dist_sq = torch.sum((x_clone - x_original_expanded)**2, dim=1)
            objective_per_sample = loss_values - LAMBDA_PARAM * dist_sq
            if method == 'WFR':
                # Reshape objective values to match weights tensor (batch_size, num_particles)
                objective_reshaped = objective_per_sample.view(batch_size, -1)
                # Calculate the weighted sum for each original point, then average over the batch
                avg_objective = (objective_reshaped * weights).sum(dim=1).mean().item()
            else:
                # For WGF and WRM, use the simple mean as weights are uniform
                avg_objective = objective_per_sample.mean().item()
            objective_history.append(avg_objective)

        x_clone.requires_grad_(True)
        loss_values_for_grad = nn.CrossEntropyLoss(reduction='none')(saa_model(x_clone), y_repeated)
        grads, = torch.autograd.grad(loss_values_for_grad, x_clone, grad_outputs=torch.ones_like(loss_values_for_grad))
        x_clone = x_clone.detach()
        
        # --- Particle Update Logic ---
        update_direction = lr * (grads - 2 * LAMBDA_PARAM * (x_clone - x_original_expanded))
        
        if method == 'WRM':
            # WRM is deterministic gradient ascent (no noise)
            x_clone += update_direction
        else: # WFR and WGF use Langevin Dynamics (with noise)
            mean = x_clone + update_direction
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

                    x_reshaped = x_clone.view(batch_size, num_samples_per_point, -1)

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

        frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
        plot_frame_samples(model=saa_model, X=X_train, y=y_train, X_perturbed=x_clone, title=f"SAA Boundary with {method} Perturbations", save_path=frame_path, method="SAA", epoch=f"Step: {step+1}", weights=weights)
        gif_filenames.append(frame_path)

    if imageio:
        print(f"Creating SAA {method} perturbation GIF...")
        gif_path = os.path.join(output_dir, f"saa_perturbation_{method}.gif")
        with imageio.get_writer(gif_path, mode='I', fps=200, loop=0, format='gif') as writer:
            for filename in gif_filenames:
                writer.append_data(imageio.imread(filename))
        print(f"GIF saved to {gif_path}")
        
    return objective_history

# --- 5. NEW Plotting function for Objective Comparison ---S
def plot_objective_comparison(histories: Dict[str, list], save_path_png: str):
    plt.style.use('jz.mplstyle') # Use a style that is likely to be available
    plt.figure(figsize=(3, 2.613), dpi=300, tight_layout=True)
    plt.rcParams['mathtext.fontset'] = 'cm'
    for name, history in histories.items():
        plt.plot(history, label=name, linewidth=2)
        
    plt.xlabel("iteration")
    plt.ylabel(r"loss") #$\mathbb{E}_{x\sim \hat{\mathbb{P}}}\mathbb{E}_{z\sim \hat{\mathbb{P}}_{\theta,x}}[\tilde{f}_{\theta, \lambda, x}(z)]$
    # plt.title("Objective Values Variation")
    plt.legend()
    plt.grid(False)

    save_path_pdf = save_path_png.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    plt.close()
    print(f"Objective comparison plot saved to {save_path_png} and {save_path_pdf}")

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
    plt.style.use('default') # Use a default style that is likely to be available
    X_train, y_train = create_dataset(n_samples=N_SAMPLES_TRAIN, noise=0.1, imbalance_ratio=0.8, random_state=SEED)
    X_test, y_test = create_dataset(n_samples=N_SAMPLES_TEST, noise=0.3, random_state=SEED+1)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_test_torch = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_torch = torch.from_numpy(y_test).long().to(DEVICE)

    # --- Models & History Logging---
    true_boundary_model = get_true_boundary_model(INPUT_DIM)

    LOG_EPOCH_INTERVAL = 10
    ROBUSTNESS_LOG_INTERVAL = 10

    model_names = ["SAA", "SVGD", "MMD", "Dual", "WGF", "WRM", "WFR"]
    loss_histories = {name: [] for name in ["SVGD", "Dual", "WGF", "WFR", "MMD"]}

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

    # ==========================================================================
    # =========================== MODIFIED SECTION =============================
    # ==========================================================================
    # Call the force visualization functions for all relevant methods
    plot_perturbation_process_kernel_force(saa_model, X_train, y_train, 'SVGD', OUTPUT_DIR, lr=1e-2, inner_steps=200)
    wgf_forces_history = plot_perturbation_process_wasserstein_forces(saa_model, X_train, y_train, 'WGF', OUTPUT_DIR, lr=inner_lr, inner_steps=200)
    wfr_forces_history = plot_perturbation_process_wasserstein_forces(saa_model, X_train, y_train, 'WFR', OUTPUT_DIR, lr=inner_lr, inner_steps=200)
    wrm_forces_history = plot_perturbation_process_wasserstein_forces(saa_model, X_train, y_train, 'WRM', OUTPUT_DIR, lr=inner_lr, inner_steps=200)
    
    # NEW: Call the kernel force visualization function
    plot_perturbation_process_kernel_force(saa_model, X_train, y_train, 'MMD', OUTPUT_DIR, lr=inner_lr, inner_steps=200)
    


    # Plot objective value comparison for Wasserstein-based methods
    comparison_histories = {
        "WFR (Forces Viz)": wfr_forces_history,
        "WGF (Forces Viz)": wgf_forces_history,
        "WRM (Forces Viz)": wrm_forces_history
    }
    plot_objective_comparison(
        comparison_histories,
        os.path.join(OUTPUT_DIR, "saa_perturb_objective_comparison_forces.png")
    )
    # ==========================================================================
    # ========================= END MODIFIED SECTION ===========================
    # ==========================================================================


    # --- DRO Models Training ---
    dro_methods_to_train = ["SVGD", "MMD", "WFR", "WRM", "WGF", "Dual"]

    for method in dro_methods_to_train:
        print(f"\n--- Training {method} Model ---")
        model = Classifier(INPUT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        # MODIFICATION: Create subdirectories for PNG and PDF frames
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
                    #record_loss = compute_sinkhorn_loss(model(sinkhorn_base_sampler(X_batch, m_samples=NUM_SAMPLES_PER_POINT)), y_repeated, NUM_SAMPLES_PER_POINT, LAMBDA_PARAM * EPSILON)
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
                    #record_loss = compute_sinkhorn_loss(model(sinkhorn_base_sampler(X_batch, m_samples=NUM_SAMPLES_PER_POINT)), y_repeated, NUM_SAMPLES_PER_POINT, LAMBDA_PARAM * EPSILON)
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

            # MODIFIED: Evaluate with adversarial attacks
            if (epoch + 1) % ROBUSTNESS_LOG_INTERVAL == 0:
                model.eval()
                X_test_fgsm = fgsm_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON)
                X_test_pgd = pgd_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON, PGD_ALPHA, PGD_ITERS)

                acc_fgsm = evaluate_accuracy(model, X_test_fgsm, y_test_torch)
                acc_pgd = evaluate_accuracy(model, X_test_pgd, y_test_torch)
                robustness_history[method]["FGSM"].append(acc_fgsm)
                robustness_history[method]["PGD"].append(acc_pgd)

            if method in ["SVGD", "WGF", "WRM", "WFR", "MMD"]:
                model.eval()
                epoch_weights = None # Default to None for methods without weights
                if method == "WGF":
                    epoch_X_p = wgf_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "WRM":
                    epoch_X_p = wrm_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "WFR":
                    # Capture weights from the sampler for visualization
                    epoch_X_p, epoch_weights = wfr_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "MMD":
                    epoch_X_p = mmd_dro_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch)
                elif method == "SVGD":
                    epoch_X_p = svgd_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)
                # --- MODIFIED: Save frames in subdirectory ---
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

    # --- Final Visualizations and Evaluations ---
    print("\n--- Generating Final Plots and Evaluations ---")

    # Call the new function for WFR weight evolution analysis


    # 1. Loss Comparison Plot
    plot_loss_comparison(loss_histories, f"{OUTPUT_DIR}/loss_comparison.png")
    print(f"Loss comparison plot saved to {OUTPUT_DIR}/loss_comparison.png and {OUTPUT_DIR_PDF}/loss_comparison.pdf")

    # 2. Boundary Comparison Plot
    visualize_all_boundaries(trained_models, true_boundary_model, X_train, y_train, "Decision Boundaries", f"{OUTPUT_DIR}/final_boundary_comparison.png")
    print(f"Final boundary comparison plot saved to {OUTPUT_DIR}/final_boundary_comparison.png and {OUTPUT_DIR_PDF}/final_boundary_comparison.pdf")

    # 3. Robustness Evolution Plot
    plot_robustness_comparison(robustness_history, OUTPUT_DIR, OUTPUT_DIR_PDF)

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

    print(f"\nExperiment complete. All results saved in '{OUTPUT_DIR}' and '{OUTPUT_DIR_PDF}' directories.")