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
from matplotlib.colors import to_rgb

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
N_SAMPLES_TRAIN = 200
N_SAMPLES_TEST = 1000
INPUT_DIM = 2
HIDDEN_DIM = 32
OUTPUT_DIM = 2
MAX_EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-2
SEED = 100

# DRO parameters
EPSILON = 0.01
LAMBDA_PARAM = 8
NUM_SAMPLES_PER_POINT = 5
SINKHORN_SAMPLE_LEVEL = 4
inner_lr = 1e-2
# MODIFIED: Reduced inner steps for warm start efficiency
inner_steps_warm_start = 20
inner_steps_cold_start = 100
wfr_times = 8

# Attack parameters
ATTACK_EPSILON = 0.3
PGD_ALPHA = 0.05
PGD_ITERS = 5

# Setup output directory
OUTPUT_DIR = "two_moons_results_warm_start"
OUTPUT_DIR_PDF = "two_moons_results_warm_start_pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_PDF, exist_ok=True)
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

# --- 3. DRO Samplers ---
# Note: RGO, WRM and other non-Langevin samplers remain unchanged
def rgo_sampler(x_original_batch, y_original_batch, model, epoch, inner_lr=inner_lr, inner_steps=inner_steps_cold_start):
    batch_size = x_original_batch.shape[0]
    x_pert_batch = x_original_batch.clone().detach()
    steps = int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))
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

def wrm_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps_cold_start):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    for _ in range(int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))):
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_original_batch)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone +  lr * (grads  - 2*LAMBDA_PARAM*(x_clone - x_original_batch))
    return x_clone.detach()

# --- NEW: Warm Start Samplers ---
def wgf_sampler_warm_start(x_original_batch, y_original_batch, model, initial_particles, lr=inner_lr, inner_steps=inner_steps_warm_start):
    """ WGF sampler that starts from initial_particles (Warm Start). """
    x_clone = initial_particles.clone().detach()
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).reshape(-1, INPUT_DIM)

    for _ in range(inner_steps):
        x_clone.requires_grad_(True)
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone.detach()

        mean = x_clone + lr * (grads - 2 * LAMBDA_PARAM * (x_clone - x_original_expanded))
        std_dev = torch.sqrt(torch.tensor(2 * lr * LAMBDA_PARAM * EPSILON, device=DEVICE))
        noise = torch.randn_like(mean) * std_dev
        x_clone = mean + noise

    return x_clone.detach()

def wfr_sampler_warm_start(x_original_batch, y_original_batch, model, initial_particles, initial_weights, lr=inner_lr, inner_steps=inner_steps_warm_start):
    """ WFR sampler that starts from initial_particles and initial_weights (Warm Start). """
    device = x_original_batch.device
    batch_size = x_original_batch.shape[0]

    weights = initial_weights.clone().detach()
    x_clone = initial_particles.clone().detach()
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).reshape(-1, INPUT_DIM)

    wt_lr = lr * wfr_times
    weight_exponent = 1 - LAMBDA_PARAM * EPSILON * wt_lr

    for _ in range(inner_steps):
        x_clone.requires_grad_(True)
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone.detach()

        mean = x_clone + lr * (grads - 2 * LAMBDA_PARAM * (x_clone - x_original_expanded))
        std_dev = torch.sqrt(torch.tensor(2 * lr * LAMBDA_PARAM * EPSILON, device=device))
        noise = torch.randn_like(mean) * std_dev
        x_clone = mean + noise

        with torch.no_grad():
            new_loss = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated).view(batch_size, -1) - 2 * LAMBDA_PARAM * torch.sum((x_clone - x_original_expanded) ** 2, dim=1).view(batch_size, -1)
            weights = weights ** weight_exponent * torch.exp(new_loss * wt_lr)
            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

    return x_clone.detach(), weights.detach()


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
def plot_frame_samples(model, X, y, X_perturbed, title, save_path, method, epoch, weights=None):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)

    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5, s=20, linewidths=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5, s=20, linewidths=0.5)

    if X_perturbed is not None:
        X_p_np = X_perturbed.cpu().numpy()
        
        if weights is None:
            num_repeats = X_p_np.shape[0] // X.shape[0]
            weights = torch.ones((X.shape[0], num_repeats), device=X_perturbed.device) / num_repeats

        if X_p_np.shape[0] != weights.numel():
            print(f"Warning in plot_frame_samples: Mismatch between perturbed points ({X_p_np.shape[0]}) and weights ({weights.numel()}). Truncating to match.")
            min_len = min(X_p_np.shape[0], weights.numel())
            X_p_np = X_p_np[:min_len]
            weights = weights.flatten()[:min_len].reshape(X.shape[0], -1)

        num_repeats = weights.shape[1]
        y_repeated = np.repeat(y, num_repeats)

        weights_flat = weights.flatten().cpu().numpy()

        min_alpha, max_alpha = 0.05, 0.6
        w_min, w_max = weights_flat.min(), weights_flat.max()
        if w_max > w_min:
            alphas = min_alpha + (weights_flat - w_min) / (w_max - w_min) * (max_alpha - min_alpha)
        else:
            alphas = np.full_like(weights_flat, max_alpha)

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
                ax.plot([original_point[0], p_point[0]], [original_point[1], p_point[1]], color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        Z_model = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)

    ax.contour(xx, yy, Z_model, levels=[0.5], linewidths=1.5, colors='black', linestyles='--')

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', lw=2, label=f'Decision Boundary ({method})'),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', alpha=0.6, label='Perturbed Positive', markersize=5, markeredgecolor="gray", markeredgewidth=0.2),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', alpha=0.6, label='Perturbed Negative', markersize=5, markeredgecolor="gray", markeredgewidth=0.2)
    ]

    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.text(0.95, 0.05, f'Epoch: {epoch}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.set_title(title, fontsize=9)
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
    colors = ["#E2E60C", "#C9151E", "#3A1FB4", '#CC6677', "#951699", "#1C7414EF"]

    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linewidths=1.5)

    ideal_color = 'black'
    ideal_linewidth = 1.5
    center1 = (0, 0.25); center2 = (1, 0.25); radius = 0.5
    arc1 = Arc(center1, width=radius*2, height=radius*2, theta1=0, theta2=180, edgecolor=ideal_color, lw=ideal_linewidth, linestyle='-')
    ax.add_patch(arc1)
    arc2 = Arc(center2, width=radius*2, height=radius*2, theta1=180, theta2=360, edgecolor=ideal_color, lw=ideal_linewidth, linestyle='-')
    ax.add_patch(arc2)
    ax.plot([-0.5, -0.5], [0.25, y_min], color=ideal_color, lw=ideal_linewidth, linestyle='-')
    ax.plot([1.5, 1.5], [0.25, y_max], color=ideal_color, lw=ideal_linewidth, linestyle='-')

    legend_elements = [
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive', markersize=4, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative', markersize=4, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="Ideal Boundary")
    ]
    legend_elements.extend([Line2D([0], [0], lw=1.5, color=colors[i], label=name) for i, name in enumerate(models.keys())])
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1)); ax.set_title(title)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    
    save_path_pdf = save_path_png.replace(OUTPUT_DIR, OUTPUT_DIR_PDF).replace('.png', '.pdf')
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    plt.close(fig)

# --- 5. Attack and Evaluation Functions ---
def fgsm_attack(model, loss_fn, data, labels, epsilon):
    data_adv = data.clone().detach().requires_grad_(True)
    outputs = model(data_adv)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad = data_adv.grad.data
    data_adv = data_adv.detach() + epsilon * grad.sign()
    return data_adv

def pgd_attack(model, loss_fn, data, labels, epsilon, alpha, iters):
    data_adv = data.clone().detach() + torch.empty_like(data).uniform_(-epsilon, epsilon)
    for _ in range(iters):
        data_adv.requires_grad_(True)
        outputs = model(data_adv)
        loss = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = data_adv.grad.data
        data_adv = data_adv.detach() + alpha * grad.sign()
        eta = torch.clamp(data_adv - data, min=-epsilon, max=epsilon)
        data_adv = data.clone().detach() + eta
    return data_adv

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
    plt.style.use('default')
    X_train, y_train = create_dataset(n_samples=N_SAMPLES_TRAIN, noise=0.1, imbalance_ratio=0.9, random_state=SEED)
    X_test, y_test = create_dataset(n_samples=N_SAMPLES_TEST, noise=0.3, random_state=SEED+1)
    
    # --- NEW: Create a dataset and loader that includes indices for warm-starting ---
    X_train_torch = torch.from_numpy(X_train).float()
    y_train_torch = torch.from_numpy(y_train).long()
    train_indices = torch.arange(len(X_train))
    train_dataset_with_indices = TensorDataset(X_train_torch, y_train_torch, train_indices)
    train_loader_with_indices = DataLoader(train_dataset_with_indices, batch_size=BATCH_SIZE, shuffle=True)
    
    # Original loader for methods that don't use warm-start
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_test_torch = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_torch = torch.from_numpy(y_test).long().to(DEVICE)

    # --- Models & History Logging ---
    LOG_EPOCH_INTERVAL = 10
    ROBUSTNESS_LOG_INTERVAL = 10
    model_names = ["SAA", "RGO", "Dual", "WGF", "WRM", "WFR"]
    robustness_history = {name: {"FGSM": [], "PGD": []} for name in model_names}
    criterion = nn.CrossEntropyLoss()
    trained_models = {}

    # --- SAA Training (Standard Model) ---
    print("\n--- Training Standard (SAA) Model ---")
    saa_model = Classifier(INPUT_DIM).to(DEVICE)
    optimizer_saa = optim.Adam(saa_model.parameters(), lr=LR)
    for epoch in tqdm(range(MAX_EPOCHS), desc="SAA Training"):
        saa_model.train()
        for X_batch, y_batch in train_loader:
            optimizer_saa.zero_grad()
            loss = criterion(saa_model(X_batch.to(DEVICE)), y_batch.to(DEVICE))
            loss.backward()
            optimizer_saa.step()
    trained_models["SAA"] = saa_model

    # --- DRO Models Training ---
    dro_methods_to_train = {
        "WFR (Warm Start)": "WFR",
        "WGF (Warm Start)": "WGF",
        "RGO": "RGO",
        "WRM": "WRM",
        "Dual": "Dual"
    }

    for method_name, method_key in dro_methods_to_train.items():
        print(f"\n--- Training {method_name} Model ---")
        model = Classifier(INPUT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        method_frames_dir = os.path.join(OUTPUT_DIR, f"{method_key}_frames")
        os.makedirs(method_frames_dir, exist_ok=True)
        gif_filenames = []

        # --- MODIFIED: Warm Start Setup for WGF and WFR ---
        is_warm_start = method_key in ["WGF", "WFR"]
        if is_warm_start:
            # State for perturbed particle positions
            perturbed_samples_state = X_train_torch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).contiguous().view(-1, INPUT_DIM).to(DEVICE)
            perturbed_samples_state += torch.randn_like(perturbed_samples_state) * 0.01 # Add small initial noise

            if method_key == "WFR":
                weights_state = torch.ones((N_SAMPLES_TRAIN, NUM_SAMPLES_PER_POINT), dtype=torch.float32, device=DEVICE) / NUM_SAMPLES_PER_POINT
        
        for epoch in tqdm(range(MAX_EPOCHS), desc=f"{method_name} Training"):
            model.train()
            
            # --- MODIFIED: Select the correct data loader and training loop ---
            if is_warm_start:
                current_loader = train_loader_with_indices
                for X_batch, y_batch, indices in current_loader:
                    X_batch, y_batch, indices = X_batch.to(DEVICE), y_batch.to(DEVICE), indices.to(DEVICE)
                    model.eval()
                    
                    # Flatten indices to get particle locations from the state tensor
                    particle_indices_flat = (indices.view(-1, 1) * NUM_SAMPLES_PER_POINT + torch.arange(NUM_SAMPLES_PER_POINT, device=DEVICE)).view(-1)
                    initial_particles = perturbed_samples_state[particle_indices_flat]
                    
                    if method_key == "WGF":
                        X_perturbed = wgf_sampler_warm_start(X_batch, y_batch, model, initial_particles=initial_particles)
                        y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                        model.train()
                        loss = criterion(model(X_perturbed), y_repeated)
                        perturbed_samples_state[particle_indices_flat] = X_perturbed.detach() # Update state

                    elif method_key == "WFR":
                        initial_weights = weights_state[indices]
                        X_perturbed, weights = wfr_sampler_warm_start(X_batch, y_batch, model, initial_particles=initial_particles, initial_weights=initial_weights)
                        y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                        model.train()
                        predictions = model(X_perturbed)
                        loss_values = nn.CrossEntropyLoss(reduction='none')(predictions, y_repeated)
                        loss = (loss_values.view(-1, NUM_SAMPLES_PER_POINT) * weights).sum(dim=1).mean()
                        
                        perturbed_samples_state[particle_indices_flat] = X_perturbed.detach()
                        weights_state[indices] = weights.detach()

                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            else: # Original loop for other methods
                current_loader = train_loader
                for X_batch, y_batch in current_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    
                    if method_key == "Dual":
                        m = 2 ** np.random.choice(np.arange(SINKHORN_SAMPLE_LEVEL + 1), p=(2.**-np.arange(SINKHORN_SAMPLE_LEVEL + 1))/(2. - 2.**-SINKHORN_SAMPLE_LEVEL))
                        X_perturbed = sinkhorn_base_sampler(X_batch, m_samples=m)
                        y_repeated = y_batch.repeat_interleave(m); predictions = model(X_perturbed)
                        loss = compute_sinkhorn_loss(predictions, y_repeated, m, LAMBDA_PARAM * EPSILON)
                    
                    elif method_key == "RGO":
                        model.eval()
                        X_perturbed = rgo_sampler(X_batch, y_batch, model, epoch)
                        y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                        model.train()
                        loss = criterion(model(X_perturbed), y_repeated)
                    
                    elif method_key == "WRM":
                        model.eval()
                        X_perturbed = wrm_sampler(X_batch, y_batch, model, epoch)
                        loss = criterion(model(X_perturbed), y_batch)

                    optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Visualization logic
            model.eval()
            epoch_weights = None
            if is_warm_start:
                epoch_X_p = perturbed_samples_state.view(-1, INPUT_DIM)
                if method_key == "WFR":
                    epoch_weights = weights_state
            elif method_key == "RGO":
                epoch_X_p = rgo_sampler(X_train_torch.to(DEVICE), y_train_torch.to(DEVICE), model, epoch)
            elif method_key == "WRM":
                # For WRM, we visualize a single perturbed point per original point
                epoch_X_p = wrm_sampler(X_train_torch.to(DEVICE), y_train_torch.to(DEVICE), model, epoch)
            else: # For Dual, we don't generate a persistent perturbed set
                epoch_X_p = None

            samples_frame_path = os.path.join(method_frames_dir, f"{method_key}_frame_{epoch:03d}_samples.png")
            plot_frame_samples(model, X_train, y_train, epoch_X_p, f"Worst-Case Sample Distribution ({method_name})", samples_frame_path, method_name, epoch, weights=epoch_weights)
            gif_filenames.append(samples_frame_path)

        trained_models[method_name] = model

        if imageio and gif_filenames:
            print(f"Creating GIF for {method_name}...")
            gif_path = f"{OUTPUT_DIR}/{method_key}_evolution_samples.gif"
            with imageio.get_writer(gif_path, mode='I', duration=200, loop=0, format='gif') as writer:
                for filename in gif_filenames:
                    writer.append_data(imageio.imread(filename))
            print(f"GIF saved to {gif_path}")

    # --- Final Visualizations and Evaluations ---
    print("\n--- Generating Final Plots and Evaluations ---")

    visualize_all_boundaries(trained_models, X_train, y_train, "Decision Boundary Comparison of All Models", f"{OUTPUT_DIR}/final_boundary_comparison.png")
    print(f"Final boundary comparison plot saved.")

    print("\n--- Evaluating Final Model Robustness ---")
    print("-" * 80)
    print(f"Attack Strength (Epsilon): {ATTACK_EPSILON}")
    print("-" * 80)
    print(f"| {'Model':<20} | {'Clean Accuracy':<15} | {'FGSM Accuracy':<15} | {'PGD Accuracy':<15} |")
    print("-" * 80)
    for name, model in trained_models.items():
        model.eval()
        X_test_fgsm = fgsm_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON)
        X_test_pgd = pgd_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON, PGD_ALPHA, PGD_ITERS)
        acc_clean = evaluate_accuracy(model, X_test_torch, y_test_torch)
        acc_fgsm = evaluate_accuracy(model, X_test_fgsm, y_test_torch)
        acc_pgd = evaluate_accuracy(model, X_test_pgd, y_test_torch)
        print(f"| {name:<20} | {acc_clean:^15.2%} | {acc_fgsm:^15.2%} | {acc_pgd:^15.2%} |")
    print("-" * 80)

    print(f"\nExperiment complete. All results saved in '{OUTPUT_DIR}' and '{OUTPUT_DIR_PDF}' directories.")

