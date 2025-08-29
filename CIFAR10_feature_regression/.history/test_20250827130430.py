import torch
import torch.nn as nn
import numpy as np
import os
import math
from typing import Tuple, List

# Matplotlib/Seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib.patches import Patch

# Attempt to import imageio for GIF creation
try:
    import imageio
except ImportError:
    print("The 'imageio' library is not installed. GIF generation will be skipped.")
    print("Please install it to create GIFs: pip install imageio")
    imageio = None

# --- 1. Configuration and Global Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
DATA_FILE = "cifar10_resnet50_features.pth"
PLOT_DIR = "perturbation_plots_multi_sample"
SEED = 2024

# --- Parameters for Data Sub-sampling ---
NUM_CLASSES_TO_SELECT = 2
NUM_SAMPLES_PER_CLASS = 100
NUM_PERTURBATIONS_PER_SAMPLE = 10

# --- Parameters as per your request ---
INPUT_DIM = 250
TOTAL_CLASSES = 10
LAMBDA_PARAM = 10
EPSILON = 0.01
MAX_EPOCHS = 30

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 2. Model and DRO Class Definitions ---
class DROError(Exception): pass

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.linear(x)

class BaseLinearDRO:
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool, sample_level: int = 6):
        self.input_dim, self.num_classes, self.fit_intercept = input_dim, num_classes, fit_intercept
        self.device = torch.device("cpu")
        self.model: nn.Module
        self.sample_level = sample_level
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor: return torch.as_tensor(data, dtype=torch.float32, device=self.device)

class SinkhornDROLogisticRGO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 20, num_samples: int = 1, max_iter: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.rgo_inner_lr, self.rgo_inner_steps = rgo_inner_lr, rgo_inner_steps
        self.num_samples, self.max_iter, self.learning_rate, self.batch_size = num_samples, max_iter, learning_rate, batch_size
        self.rgo_vectorized_max_trials = 100
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _rgo_sampler_vectorized(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor, current_model_state: nn.Module, num_samples_to_generate: int, epoch: int) -> torch.Tensor:
        batch_size = x_original_batch.size(0)
        x_orig_detached_batch = x_original_batch.detach()
        x_pert_batch = x_orig_detached_batch.clone()
        lr_inner = self.rgo_inner_lr
        inner_steps = int(min(5, self.rgo_inner_steps * (epoch + 1) / self.max_iter))
        for _ in range(inner_steps):
            x_pert_batch.requires_grad_(True)
            per_sample_losses = self._get_model_loss_value_batched(x_pert_batch, y_original_batch, current_model_state)
            per_sample_grads, = torch.autograd.grad(outputs=per_sample_losses, inputs=x_pert_batch, grad_outputs=torch.ones_like(per_sample_losses))
            x_pert_batch = x_pert_batch.detach()
            grad_total = -per_sample_grads / self.lambda_param + 2 * (x_pert_batch - x_orig_detached_batch)
            x_pert_batch -= lr_inner * grad_total
        x_opt_star_batch = x_pert_batch
        var_rgo = self.epsilon
        if var_rgo <= 1e-12: return x_opt_star_batch.repeat_interleave(num_samples_to_generate, dim=0)
        std_rgo = math.sqrt(var_rgo)
        f_model_loss_opt_star = self._get_model_loss_value_batched(x_opt_star_batch, y_original_batch, current_model_state)
        norm_sq_opt_star = torch.sum((x_opt_star_batch - x_orig_detached_batch) ** 2, dim=1)
        f_L_xi_opt_star = (-f_model_loss_opt_star / (self.lambda_param * self.epsilon)) + (norm_sq_opt_star / self.epsilon)
        x_opt_star_3d, x_original_3d, f_L_xi_opt_star_3d = x_opt_star_batch.unsqueeze(1), x_orig_detached_batch.unsqueeze(1), f_L_xi_opt_star.unsqueeze(1)
        final_accepted_perturbations = torch.zeros((batch_size, num_samples_to_generate, self.input_dim), device=self.device)
        active_flags = torch.ones((batch_size, num_samples_to_generate), dtype=torch.bool, device=self.device)
        for _ in range(self.rgo_vectorized_max_trials):
            if not active_flags.any(): break
            pert_proposals = torch.randn_like(final_accepted_perturbations) * std_rgo
            x_candidates = x_opt_star_3d + pert_proposals
            x_candidates_flat = x_candidates.view(-1, self.input_dim)
            y_repeated = y_original_batch.repeat_interleave(num_samples_to_generate, dim=0)
            f_model_loss_candidates = self._get_model_loss_value_batched(x_candidates_flat, y_repeated, current_model_state).view(batch_size, num_samples_to_generate)
            norm_sq_candidates = torch.sum((x_candidates - x_original_3d) ** 2, dim=2)
            f_L_xi_candidates = (-f_model_loss_candidates / (self.lambda_param * self.epsilon)) + (norm_sq_candidates / self.epsilon)
            diff_cand_opt_norm_sq = torch.sum(pert_proposals**2, dim=2)
            exponent_term3 = diff_cand_opt_norm_sq / (2 * var_rgo)
            acceptance_probs = torch.exp(torch.clamp(-f_L_xi_candidates + f_L_xi_opt_star_3d + exponent_term3, max=10))
            newly_accepted_mask = (torch.rand_like(acceptance_probs) < acceptance_probs) & active_flags
            final_accepted_perturbations[newly_accepted_mask] = pert_proposals[newly_accepted_mask]
            active_flags[newly_accepted_mask] = False
        return (x_opt_star_3d + final_accepted_perturbations).view(-1, self.input_dim)

class SinkhornDROLD(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 1, max_iter: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.num_samples, self.max_iter, self.learning_rate, self.batch_size = num_samples, max_iter, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _LD_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module) -> torch.Tensor:
        batch_size = x_orig.size(0)
        x_orig_clone = x_orig.clone().detach().requires_grad_(True)
        predictions = model(x_orig_clone)
        loss = nn.CrossEntropyLoss(reduction='sum')(predictions, y_orig)
        loss.backward()
        grad = x_orig_clone.grad.data
        mean = x_orig + grad * self.epsilon / 2
        std_dev = torch.sqrt(torch.tensor(self.epsilon, device=self.device))
        mean_expanded = mean.unsqueeze(1).expand(batch_size, self.num_samples, self.input_dim)
        noise = std_dev * torch.randn(batch_size, self.num_samples, self.input_dim, device=self.device)
        X0_torch = mean_expanded + noise
        X0_torch = X0_torch.view(-1, self.input_dim)
        return X0_torch

# --- 3. Main Execution Logic ---
if __name__ == '__main__':
    # --- Setup Directories ---
    os.makedirs(PLOT_DIR, exist_ok=True)
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Checkpoint directory '{CHECKPOINT_DIR}' not found. Please run the training script first.")
        exit()
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found.")
        exit()

    # --- Load and Sub-sample Data ---
    print("Loading data and sub-sampling...")
    data = torch.load(DATA_FILE)
    X_full = data["train_features"].numpy()
    if X_full.shape[1] < INPUT_DIM:
        print(f"Error: Loaded features have dimension {X_full.shape[1]}, which is less than the required INPUT_DIM={INPUT_DIM}.")
        exit()
    X_full = X_full[:, :INPUT_DIM]
    y_full = data["train_labels"].numpy()

    all_classes = np.unique(y_full)
    selected_classes = np.random.choice(all_classes, NUM_CLASSES_TO_SELECT, replace=False)
    selected_classes.sort()
    print(f"Selected classes for visualization: {selected_classes}")

    X_sample_list, y_sample_list = [], []
    for cls in selected_classes:
        class_indices = np.where(y_full == cls)[0]
        sample_indices = np.random.choice(class_indices, NUM_SAMPLES_PER_CLASS, replace=False)
        X_sample_list.append(X_full[sample_indices])
        y_sample_list.append(y_full[sample_indices])

    X_sample = np.vstack(X_sample_list)
    y_sample = np.concatenate(y_sample_list)
    X_sample_torch = torch.from_numpy(X_sample).to(DEVICE)
    y_sample_torch = torch.from_numpy(y_sample).long().to(DEVICE)
    
    # --- ‼️ Step 1: Pre-compute all perturbed data across all epochs ‼️ ---
    print("Pre-computing all perturbations to determine global axis limits...")
    methods_to_visualize = {
        "RGO": (SinkhornDROLogisticRGO, "SinkhornDROLogisticRGO"),
        "LD": (SinkhornDROLD, "SinkhornDROLD")
    }
    all_perturbed_data = { "RGO": [], "LD": [] }

    for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc="Pre-computing data"):
        for method_name, (ModelClass, file_prefix) in methods_to_visualize.items():
            model_path = os.path.join(CHECKPOINT_DIR, f"{file_prefix}_epoch_{epoch}.pth")
            if not os.path.exists(model_path):
                all_perturbed_data[method_name].append(None) # Placeholder if model is missing
                continue

            dro_instance = ModelClass(
                input_dim=INPUT_DIM, num_classes=TOTAL_CLASSES, device=DEVICE.type,
                lambda_param=LAMBDA_PARAM, epsilon=EPSILON, num_samples=NUM_PERTURBATIONS_PER_SAMPLE
            )
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            dro_instance.model.load_state_dict(state_dict)
            dro_instance.model.eval()

            if method_name == "RGO":
                X_perturbed_torch = dro_instance._rgo_sampler_vectorized(X_sample_torch, y_sample_torch, dro_instance.model, NUM_PERTURBATIONS_PER_SAMPLE, epoch)
            else:
                X_perturbed_torch = dro_instance._LD_sampler(X_sample_torch, y_sample_torch, dro_instance.model)
            
            all_perturbed_data[method_name].append(X_perturbed_torch.cpu().detach().numpy())

    # --- ‼️ Step 2: Fit PCA on the entire dataset (original + all perturbations) ‼️ ---
    print("Fitting PCA on the complete dataset...")
    all_points_for_pca = [X_sample] + [p for p_list in all_perturbed_data.values() for p in p_list if p is not None]
    full_dataset = np.vstack(all_points_for_pca)
    
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(full_dataset)

    # Transform all data to 2D
    X_sample_2d = pca.transform(X_sample)
    all_perturbed_2d = { "RGO": [], "LD": [] }
    for method in methods_to_visualize.keys():
        for p_data in all_perturbed_data[method]:
            if p_data is not None:
                all_perturbed_2d[method].append(pca.transform(p_data))
            else:
                all_perturbed_2d[method].append(None)

    # --- ‼️ Step 3: Calculate global axis boundaries ‼️ ---
    all_2d_points = np.vstack([X_sample_2d] + [p for p_list in all_perturbed_2d.values() for p in p_list if p is not None])
    min_x, max_x = all_2d_points[:, 0].min(), all_2d_points[:, 0].max()
    min_y, max_y = all_2d_points[:, 1].min(), all_2d_points[:, 1].max()
    
    # Add a 10% margin
    margin_x = (max_x - min_x) * 0.1
    margin_y = (max_y - min_y) * 0.1
    global_xlim = (min_x - margin_x, max_x + margin_x)
    global_ylim = (min_y - margin_y, max_y + margin_y)
    print(f"Global axis limits set to: X={global_xlim}, Y={global_ylim}")

    # --- ‼️ Step 4: Generate plots with fixed axes ‼️ ---
    png_filenames = { "RGO": [], "LD": [] }
    for epoch_idx, epoch in enumerate(tqdm(range(1, MAX_EPOCHS + 1), desc="Generating plots")):
        for method_name in methods_to_visualize.keys():
            X_perturbed_2d = all_perturbed_2d[method_name][epoch_idx]
            if X_perturbed_2d is None: continue

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.set_theme(style="whitegrid")
            palette = sns.color_palette("deep", n_colors=NUM_CLASSES_TO_SELECT)

            for i, cls in enumerate(selected_classes):
                mask_orig = (y_sample == cls)
                mask_pert = np.repeat(mask_orig, NUM_PERTURBATIONS_PER_SAMPLE)
                ax.scatter(
                    X_sample_2d[mask_orig, 0], X_sample_2d[mask_orig, 1],
                    color=palette[i], s=50, edgecolor='k', linewidth=0.5, zorder=3
                )
                sns.kdeplot(
                    x=X_perturbed_2d[mask_pert, 0], y=X_perturbed_2d[mask_pert, 1],
                    ax=ax, color=palette[i], fill=True, alpha=0.3, levels=5, zorder=2
                )

            # Apply global limits
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            
            ax.set_title(f"{method_name} Perturbation Distribution (Epoch {epoch})", fontsize=16)
            ax.set_xlabel("Principal Component 1", fontsize=12)
            ax.set_ylabel("Principal Component 2", fontsize=12)
            
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label=f'Class {cls} (Original)', markerfacecolor=palette[i], markersize=10)
                for i, cls in enumerate(selected_classes)
            ]
            legend_elements.append(Patch(facecolor='gray', edgecolor='gray', alpha=0.4, label='Perturbed Distribution (KDE)'))
            ax.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            plot_filename = os.path.join(PLOT_DIR, f"{method_name.lower()}_dist_epoch_{epoch:02d}.png")
            fig.savefig(plot_filename, dpi=100)
            plt.close(fig)
            
            png_filenames[method_name].append(plot_filename)

    print("\nVisualization complete. All PNGs are saved.")

    # --- Create GIFs from the generated plots ---
    if imageio:
        print("\nCreating GIFs from the generated plots...")
        for method_name, filenames in png_filenames.items():
            if filenames:
                gif_path = os.path.join(PLOT_DIR, f"{method_name}_perturbation_evolution.gif")
                with imageio.get_writer(gif_path, mode='I', duration=500, loop=0, format='gif') as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                print(f"Successfully created GIF: {gif_path}")