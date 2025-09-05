import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import math
from tqdm import tqdm
from typing import Dict

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
N_SAMPLES_TRAIN = 400
N_SAMPLES_TEST = 1000
INPUT_DIM = 2
HIDDEN_DIM = 8
OUTPUT_DIM = 2
MAX_EPOCHS = 20
BATCH_SIZE = 64
LR = 5e-2
SEED = 42

# DRO parameters
EPSILON = 0.001
LAMBDA_PARAM = 10.0
NUM_SAMPLES_PER_POINT = 8

# Attack parameters
ATTACK_EPSILON = 0.3 # Strength of the adversarial attack
PGD_ALPHA = 0.05     # Step size for PGD
PGD_ITERS = 5       # Number of iterations for PGD

# Setup output directory
OUTPUT_DIR = "circle_results_modified"
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 2. Model and Data ---
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
        is_negative = (y_intermediate == -1)
        is_first_quadrant = (X_intermediate[:, 0] > 0) & (X_intermediate[:, 1] > 0)
        neg_q1_indices = np.where(is_negative & is_first_quadrant)[0]
        num_to_remove = int(len(neg_q1_indices) * 10 / 10)
        
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

# --- 3. DRO Samplers and Loss Functions ---
def mul_ld_sampler(x_original_batch, y_original_batch, model, lr=1e-2, inner_steps=10):
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

def sinha_sampler(x_original_batch, y_original_batch, model, lr=1e-2, inner_steps=10):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    for _ in range(inner_steps):
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_original_batch)
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
        x_clone = x_clone +  lr * (grads  - 2*LAMBDA_PARAM*(x_clone - x_original_batch))
    return x_clone.detach()

def compute_sinkhorn_loss(predictions, targets, m, lambda_reg):
    criterion = nn.CrossEntropyLoss(reduction='none')
    residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
    residual_matrix = residuals.view(-1, m).T
    return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg

# --- 4. Visualization and Evaluation ---
def plot_frame(model, X, y, X_perturbed, title, save_path, method, epoch):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.3, s=15)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.3, s=15)
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
                       Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=4, markeredgecolor='k'),
                       Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=4, markeredgecolor='k'),
                       Patch(facecolor='darkorange', alpha=0.4, label='Positive Worst-Case Dist.'),
                       Patch(facecolor='dodgerblue', alpha=0.4, label='Negative Worst-Case Dist.')]
    ax.set_xlabel('feature 1', fontsize=9); ax.set_ylabel('feature 2', fontsize=9)
    ax.text(0.95, 0.05, f'Epoch: {epoch}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7); ax.set_title(title, fontsize=9); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    plt.savefig(save_path); plt.close(fig)

def plot_frame_samples(model, X, y, X_perturbed, title, save_path, method, epoch):
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)
    
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5, s=20)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5, s=20)
    
    if X_perturbed is not None:
        X_p_np = X_perturbed.cpu().numpy()
        num_repeats = X_perturbed.shape[0] // X.shape[0]
        y_repeated = np.repeat(y, num_repeats)
        
        ax.scatter(X_p_np[y_repeated==1, 0], X_p_np[y_repeated==1, 1], c='darkorange', marker='x', label='Perturbed Positive', alpha=0.3, s=20)
        ax.scatter(X_p_np[y_repeated==0, 0], X_p_np[y_repeated==0, 1], c='dodgerblue', marker='x', label='Perturbed Negative', alpha=0.3, s=20)
        
        for i in range(X.shape[0]):
            original_point = X[i]
            start_index = i * num_repeats
            end_index = start_index + num_repeats
            perturbed_points_block = X_p_np[start_index:end_index, :]
            
            for p_point in perturbed_points_block:
                ax.plot([original_point[0], p_point[0]],
                        [original_point[1], p_point[1]],
                        color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        Z_model = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    
    ax.contour(xx, yy, Z_model, levels=[0.5], linewidths=1.5, colors='black', linestyles='--')

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', lw=2, label=f'Boundary ({method})'),
        Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=5, markeredgecolor='k'),
        Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=5, markeredgecolor='k'),
        Line2D([0], [0], linestyle='None', marker='x', color='darkorange', label='Perturbed Positive', markersize=5),
        Line2D([0], [0], linestyle='None', marker='x', color='dodgerblue', label='Perturbed Negative', markersize=5)
    ]

    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.text(0.95, 0.05, f'Epoch: {epoch}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    plt.savefig(save_path)
    plt.close(fig)

# --- START of MODIFICATION ---
def visualize_all_boundaries(models: Dict[str, nn.Module], X, y, title, save_path):
    """MODIFIED to revert back to automatic color assignment for boundaries."""
    fig, ax = plt.subplots(figsize=(4.2, 2.613), dpi=300, tight_layout=True)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5)
    x_min, x_max, y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1, X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    
    # Use a colormap to automatically assign harmonious colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    linestyles = ['--', ':', '-.', '--']
    
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=linestyles[i % len(linestyles)], linewidths=2)
        
    legend_elements = [Line2D([0], [0], linestyle='None', marker='o', color='darkorange', label='Positive Data', markersize=4, markeredgecolor='k'),
                       Line2D([0], [0], linestyle='None', marker='o', color='dodgerblue', label='Negative Data', markersize=4, markeredgecolor='k'),]
    legend_elements.extend([Line2D([0], [0], color=colors[i], lw=2, linestyle=linestyles[i % len(linestyles)], label=name) for i, name in enumerate(models.keys())])
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1)); ax.set_title(title)
    ax.set_xlabel('feature 1'); ax.set_ylabel('feature 2'); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    plt.savefig(save_path); plt.close(fig)
# --- END of MODIFICATION ---


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
    attack_types = ["FGSM", "PGD"]
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
    data_adv = data.clone().detach()
    data_adv = data_adv + torch.empty_like(data_adv).uniform_(-epsilon, epsilon)
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
    plt.style.use('seaborn-v0_8-paper') 
    
    X_train, y_train = classification_SNVD20(n_samples=N_SAMPLES_TRAIN, seed=SEED, is_train=True)
    X_test, y_test = classification_SNVD20(n_samples=N_SAMPLES_TEST, seed=SEED+1, is_train=False)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_test_torch = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_torch = torch.from_numpy(y_test).long().to(DEVICE)

    LOG_EPOCH_INTERVAL = 1
    ROBUSTNESS_LOG_INTERVAL = 1

    model_names = ["Sinha", "MultiLD"]
    loss_histories = {name: [] for name in model_names}
    robustness_history = {name: {"FGSM": [], "PGD": []} for name in model_names}
    trained_models = {}
    criterion = nn.CrossEntropyLoss()

    for method in model_names:
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

                if method == "MultiLD":
                    model.eval()
                    X_perturbed = mul_ld_sampler(X_batch, y_batch, model)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    loss = criterion(model(X_perturbed), y_repeated)
                
                elif method == "Sinha":
                    model.eval()
                    X_perturbed = sinha_sampler(X_batch, y_batch, model)
                    y_repeated = y_batch
                    loss = criterion(model(X_perturbed), y_repeated)

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item()

            if epoch % LOG_EPOCH_INTERVAL == 0:
                loss_histories[method].append(epoch_loss / len(train_loader))

            if (epoch + 1) % ROBUSTNESS_LOG_INTERVAL == 0:
                model.eval()
                X_test_fgsm = fgsm_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON)
                X_test_pgd = pgd_attack(model, criterion, X_test_torch, y_test_torch, ATTACK_EPSILON, PGD_ALPHA, PGD_ITERS)

                acc_fgsm = evaluate_accuracy(model, X_test_fgsm, y_test_torch)
                acc_pgd = evaluate_accuracy(model, X_test_pgd, y_test_torch)
                robustness_history[method]["FGSM"].append(acc_fgsm)
                robustness_history[method]["PGD"].append(acc_pgd)
            
            model.eval()
            if method == "MultiLD":
                epoch_X_p = mul_ld_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)
            elif method == "Sinha":
                epoch_X_p = sinha_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)
            
            frame_path = f"{OUTPUT_DIR}/{method}_frame_{epoch:03d}.png"
            plot_frame(model, X_train, y_train, epoch_X_p, f"Worst-case Distribution({method})", frame_path, method, epoch)
            plot_frame_samples(model, X_train, y_train, epoch_X_p, f"Worst-case Samples({method})", frame_path.replace(".png", "_samples.png"), method, epoch)
            gif_filenames_dis.append(frame_path)
            gif_filenames_sam.append(frame_path.replace(".png", "_samples.png"))

        trained_models[method] = model

        if imageio:
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

    plot_loss_comparison(loss_histories, f"{OUTPUT_DIR}/loss_comparison.png")
    print(f"Loss comparison plot saved to {OUTPUT_DIR}/loss_comparison.png")

    visualize_all_boundaries(trained_models, X_train, y_train, "Decision Boundaries", f"{OUTPUT_DIR}/final_boundary_comparison.png")
    print(f"Final boundary comparison plot saved to {OUTPUT_DIR}/final_boundary_comparison.png")

    plot_robustness_comparison(robustness_history, OUTPUT_DIR)

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