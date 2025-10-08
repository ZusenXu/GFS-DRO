import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import math
from tqdm import tqdm
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import imageio


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Experiment parameters
N_SAMPLES_TRAIN = 2000
INPUT_DIM = 2
HIDDEN_DIM = 4
OUTPUT_DIM = 2
MAX_EPOCHS = 50
BATCH_SIZE = 64
LR = 5e-2
SEED = 219

# DRO parameters
EPSILON = 0.15
LAMBDA_PARAM = 0.2
NUM_SAMPLES_PER_POINT = 8

SINKHORN_SAMPLE_LEVEL = 5

OUTPUT_DIR = "circle_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Model and Data ---
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def classification_SNVD20(n_samples, seed=42, is_train=False):
    """
    Generates synthetic data.
    If is_train is True, it removes 4/5 of the negative samples in the first quadrant.
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples * 10, 2)
    norms = np.linalg.norm(X, axis=1)
    y = np.sign(norms - np.sqrt(2))

    lower_bound = np.sqrt(2) / 1.3
    upper_bound = 1.3 * np.sqrt(2)
    gap_mask = (norms < lower_bound) | (norms > upper_bound)
    X_intermediate = X[gap_mask]
    y_intermediate = y[gap_mask]

    X_filtered = X_intermediate
    y_filtered = y_intermediate

    if is_train:
        is_first_quadrant = (X_intermediate[:, 0] > 0) & (X_intermediate[:, 1] > 0)
        neg_q1_indices = np.where(is_first_quadrant)[0] #is_negative & is_up & 
        num_to_remove = int(len(neg_q1_indices))
        
        if num_to_remove > 0:
            indices_to_remove = np.random.choice(neg_q1_indices, size=num_to_remove, replace=False)
            deletion_mask = np.ones(len(X_intermediate), dtype=bool)
            deletion_mask[indices_to_remove] = False
            X_filtered = X_intermediate[deletion_mask]
            y_filtered = y_intermediate[deletion_mask]

    X_final = X_filtered[:n_samples]
    y_final_neg_pos_one = y_filtered[:n_samples]
    y_final_0_1 = ((y_final_neg_pos_one + 1) / 2).astype(int)
    return X_final, y_final_0_1

def wgf_sampler(x_original_batch, y_original_batch, model, lr=1e-2, inner_steps=25):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    x_clone = x_clone.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).reshape(-1, INPUT_DIM)
    for _ in range(inner_steps):
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        mean = x_clone +  lr * (grads  - 2*LAMBDA_PARAM*(x_clone - x_original_expanded))
        std_dev = torch.sqrt(torch.tensor(2*lr*LAMBDA_PARAM*EPSILON, device=DEVICE))
        mean_expanded = mean.unsqueeze(1).expand(-1, 1, -1)
        noise = torch.randn_like(mean_expanded) * std_dev
        x_clone = (mean_expanded + noise).view(-1, INPUT_DIM)
    return x_clone.detach()

def wfr_sampler(x_original_batch, y_original_batch, model, lr=1e-2, inner_steps=25):
    device = x_original_batch.device
    batch_size = x_original_batch.shape[0]

    weights = torch.ones((batch_size, NUM_SAMPLES_PER_POINT), dtype=torch.float32, device=device) / NUM_SAMPLES_PER_POINT

    x_clone = x_original_batch.clone().detach().unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).contiguous().view(-1, INPUT_DIM)
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1).reshape(-1, INPUT_DIM)

    wt_lr = lr * 16
    weight_exponent = 1 - LAMBDA_PARAM * EPSILON * wt_lr

    for _ in range(inner_steps):
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
            low_weight_mask = weights < 1e-3
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

def sinha_sampler(x_original_batch, y_original_batch, model, lr=1e-2, inner_steps=25):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    for _ in range(inner_steps):
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_original_batch)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone +  lr * (grads  - 2*LAMBDA_PARAM*(x_clone - x_original_batch))
    return x_clone.detach()

def sinkhorn_base_sampler(x_original_batch, m_samples):
    expanded_data = x_original_batch.repeat_interleave(m_samples, dim=0)
    noise = torch.randn_like(expanded_data) * math.sqrt(EPSILON)
    return expanded_data + noise

def compute_sinkhorn_loss(predictions, targets, m, lambda_reg):
    criterion = nn.CrossEntropyLoss(reduction='none')
    residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
    residual_matrix = residuals.view(-1, m).T
    return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg


# --- Visualization and Evaluation ---
def plot_frame_samples(model, X, y, X_perturbed, save_path, method, epoch):
    fig, ax = plt.subplots(figsize=(4.15, 2.613), dpi=300, tight_layout=True)
    
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', alpha=0.5, s=20, linewidths=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', alpha=0.5, s=20, linewidths=0.5)
    
    if X_perturbed is not None:
        X_p_np = X_perturbed.cpu().numpy()
        num_repeats = X_perturbed.shape[0] // X.shape[0]
        y_repeated = np.repeat(y, num_repeats)
        
        ax.scatter(X_p_np[y_repeated==1, 0], X_p_np[y_repeated==1, 1], c='darkorange', marker='o',edgecolors="gray", alpha=0.3, linewidths=0.2, s=15)
        ax.scatter(X_p_np[y_repeated==0, 0], X_p_np[y_repeated==0, 1], c='dodgerblue', marker='o',edgecolors="gray", alpha=0.3, linewidths=0.2, s=15)
        
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
    
    if epoch > 0:
        ax.contour(xx, yy, Z_model, levels=[0.5], linewidths=1.5, colors='black', linestyles='--')

        legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', lw=2, label=f'Boundary ({method})'),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Perturbed Positive', markersize=5, markeredgecolor="gray", markeredgewidth=0.2),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Perturbed Negative', markersize=5, markeredgecolor="gray", markeredgewidth=0.2)
        ]
        ax.text(0.95, 0.05, f'Epoch: {epoch}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    else:
        legend_elements = [
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=5, markeredgecolor='k', markeredgewidth=0.5),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Perturbed Positive', markersize=5, markeredgecolor="gray", markeredgewidth=0.2),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Perturbed Negative', markersize=5, markeredgecolor="gray", markeredgewidth=0.2)
        ]

    ax.set_xlabel('feature 1', fontsize=9)
    ax.set_ylabel('feature 2', fontsize=9)
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    plt.savefig(save_path)
    if epoch == 0:
        plt.savefig(save_path.replace('.png', '.pdf'))
    plt.close(fig)

def visualize_all_boundaries(models: Dict[str, nn.Module], X, y, title, save_path):
    """MODIFIED to add a new color for the 'Dual' method."""
    fig, ax = plt.subplots(figsize=(4.1, 2.613), dpi=300, tight_layout=True)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5)
    x_min, x_max, y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1, X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    
    colors = ['seagreen', 'purple', "red", "saddlebrown"]
    linestyles = ['--', '--', '--', '--']
    
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=linestyles[i % len(linestyles)], linewidths=2)
        
    legend_elements = [Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=4, markeredgecolor='k'),
                       Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=4, markeredgecolor='k'),]
    legend_elements.extend([Line2D([0], [0], color=colors[i], lw=2, linestyle=linestyles[i % len(linestyles)], label=name) for i, name in enumerate(models.keys())])
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))#; ax.set_title(title)
    ax.set_xlabel('feature 1'); ax.set_ylabel('feature 2'); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    plt.savefig(save_path); plt.close(fig)

# --- Main Execution ---
if __name__ == '__main__':
    plt.style.use('seaborn-v0_8-paper') 
    
    X_train, y_train = classification_SNVD20(n_samples=N_SAMPLES_TRAIN, seed=SEED, is_train=True)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model_names = ["WRM", "WGF", "WFR", "Dual"]

    trained_models = {}
    criterion = nn.CrossEntropyLoss()
    inner_lr = 1e-3
    itr = 300
    for method in model_names:
        print(f"\n--- Training {method} Model ---")
        frames_dir = os.path.join(OUTPUT_DIR, f"{method}")
        os.makedirs(frames_dir, exist_ok=True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
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
                    levels = np.arange(SINKHORN_SAMPLE_LEVEL + 1)
                    numerators = 2.0**(-levels)
                    denominator = 2.0 - 2.0**(-SINKHORN_SAMPLE_LEVEL)
                    probabilities = numerators / denominator
                    sampled_level = np.random.choice(levels, p=probabilities)
                    m = 2 ** sampled_level
                    
                    X_perturbed = sinkhorn_base_sampler(X_batch, m_samples=m)
                    y_repeated = y_batch.repeat_interleave(m)
                    predictions = model(X_perturbed)
                    
                    loss = compute_sinkhorn_loss(predictions, y_repeated, m, LAMBDA_PARAM * EPSILON)
                elif method == "WGF":
                    model.eval()
                    X_perturbed = wgf_sampler(X_batch, y_batch, model, lr=inner_lr, inner_steps=itr)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    loss = criterion(model(X_perturbed), y_repeated)
                elif method == "WFR":
                    model.eval()
                    X_perturbed, weights = wfr_sampler(X_batch, y_batch, model)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    loss_values = nn.CrossEntropyLoss(reduction='none')(model(X_perturbed), y_repeated)
                    loss = (loss_values.view(-1, NUM_SAMPLES_PER_POINT) * weights).sum(dim=1).mean()
                elif method == "WRM":
                    model.eval()
                    X_perturbed = sinha_sampler(X_batch, y_batch, model)
                    y_repeated = y_batch
                    loss = criterion(model(X_perturbed), y_repeated)

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
                if method != "Dual":
                    model.eval()
                    if method == "WGF":
                        epoch_X_p = wgf_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)
                    elif method == "WFR":
                        epoch_X_p,_ = wfr_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)
                    elif method == "WRM":
                        epoch_X_p = sinha_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)
                    
                    frame_path = f"{OUTPUT_DIR}/{method}/{method}_frame_{epoch:03d}.png"
                    plot_frame_samples(model, X_train, y_train, epoch_X_p, frame_path, method, epoch)
                    gif_filenames_sam.append(frame_path)

        trained_models[method] = model

        if imageio and method != "Dual":
            print(f"Creating {method} evolution GIF...")
            gif_path = f"{OUTPUT_DIR}/{method}_evolution.gif"
            with imageio.get_writer(gif_path, mode='I', duration=200, loop=0, format='gif') as writer:
                for filename in gif_filenames_dis:
                    writer.append_data(imageio.imread(filename))
            with imageio.get_writer(gif_path.replace(".gif", "_samples.gif"), mode='I', duration=200, loop=0, format='gif') as writer:
                for filename in gif_filenames_sam:
                    writer.append_data(imageio.imread(filename))
            print(f"GIF saved to {gif_path}")

    print("\n--- Generating Final Plots and Evaluations ---")

    visualize_all_boundaries(trained_models, X_train, y_train, "Decision Boundaries", f"{OUTPUT_DIR}/circle_boundary_comparison.pdf")
    print(f"Final boundary comparison plot saved to {OUTPUT_DIR}/circle_boundary_comparison.pdf")

    print(f"\nExperiment complete. All results saved in '{OUTPUT_DIR}' directory.")
