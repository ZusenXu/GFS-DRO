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
N_SAMPLES_TRAIN = 200
N_SAMPLES_TEST = 1000
INPUT_DIM = 2
HIDDEN_DIM = 64
OUTPUT_DIM = 2
MAX_EPOCHS = 49
BATCH_SIZE = 64
LR = 5e-3
SEED = 42

# DRO parameters
EPSILON = 0.05
LAMBDA_PARAM = 10.0
NUM_SAMPLES_PER_POINT = 12
SINKHORN_SAMPLE_LEVEL = 6

# Attack parameters
ATTACK_EPSILON = 0.5 # Strength of the noise attack

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
    X_true, y_true = make_moons(n_samples=2000, noise=0.0, random_state=42)
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
def rgo_sampler(x_original_batch, y_original_batch, model, epoch):
    x_pert_batch = x_original_batch.clone().detach()
    inner_steps = int(min(5, 20 * (epoch + 1) / MAX_EPOCHS))
    for _ in range(inner_steps):
        x_pert_batch.requires_grad_(True)
        loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_pert_batch), y_original_batch)
        grads, = torch.autograd.grad(loss_values, x_pert_batch, grad_outputs=torch.ones_like(loss_values))
        x_pert_batch = x_pert_batch.detach()
        grad_total = -grads / LAMBDA_PARAM + 2 * (x_pert_batch - x_original_batch)
        x_pert_batch -= 0.01 * grad_total
    x_opt_star_batch = x_pert_batch
    std_rgo = math.sqrt(EPSILON)
    x_opt_star_expanded = x_opt_star_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1)
    noise = torch.randn_like(x_opt_star_expanded) * std_rgo
    return (x_opt_star_expanded + noise).view(-1, INPUT_DIM)

def ld_sampler(x_original_batch, y_original_batch, model):
    x_clone = x_original_batch.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss(reduction='sum')(model(x_clone), y_original_batch)
    loss.backward()
    grad = x_clone.grad.data
    mean = x_original_batch + grad * EPSILON / 2
    std_dev = torch.sqrt(torch.tensor(EPSILON, device=DEVICE))
    mean_expanded = mean.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1)
    noise = torch.randn_like(mean_expanded) * std_dev
    return (mean_expanded + noise).view(-1, INPUT_DIM)

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
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300, tight_layout=True)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.3)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.3)
    if X_perturbed is not None:
        num_repeats = X_perturbed.shape[0] // X.shape[0]
        y_repeated = np.repeat(y, num_repeats)
        X_p_np = X_perturbed.cpu().numpy()
        sns.kdeplot(x=X_p_np[y_repeated==1, 0], y=X_p_np[y_repeated==1, 1], ax=ax, color='darkorange', fill=True, alpha=0.4)
        sns.kdeplot(x=X_p_np[y_repeated==0, 0], y=X_p_np[y_repeated==0, 1], ax=ax, color='dodgerblue', fill=True, alpha=0.4)
    x_min, x_max, y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1, X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    with torch.no_grad(): # CORRECTED
        Z_model = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    ax.contour(xx, yy, Z_model, levels=[0.5], linewidths=2, colors='black', linestyles='--')
    legend_elements = [Line2D([0], [0], color='black', linestyle='--', lw=3, label=f'Boundary({method})'), Patch(facecolor='darkorange', alpha=0.4, label='Positive Worst-Case Dist.'), Patch(facecolor='dodgerblue', alpha=0.4, label='Negative Worst-Case Dist.')]
    ax.set_xlabel('feature 1', fontsize=11); ax.set_ylabel('feature 2', fontsize=11)
    ax.text(0.95, 0.05, f'Epoch: {epoch}', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9); ax.set_title(title, fontsize=11); ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    plt.savefig(save_path); plt.close(fig)

def visualize_all_boundaries(models: Dict[str, nn.Module], true_boundary_model, X, y, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5)
    x_min, x_max, y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1, X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    linestyles = ['--', ':', '-.', '--']
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad(): # CORRECTED
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=linestyles[i % len(linestyles)], linewidths=2.5)
    legend_elements = [Line2D([0], [0], color='black', lw=3, label='True Boundary')]
    legend_elements.extend([Line2D([0], [0], color=colors[i], lw=2.5, linestyle=linestyles[i % len(linestyles)], label=name) for i, name in enumerate(models.keys())])
    ax.legend(handles=legend_elements, loc='best'); ax.set_title(title)
    plt.savefig(save_path); plt.close(fig)

def plot_loss_comparison(loss_histories: Dict[str, list], save_path):
    plt.figure(figsize=(12, 8))
    for name, history in loss_histories.items():
        epochs_logged = range(1, (len(history) * LOG_EPOCH_INTERVAL) + 1, LOG_EPOCH_INTERVAL)
        plt.plot(epochs_logged, history, label=name, marker='o',  markersize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig(save_path)
    plt.close()

def plot_robustness_comparison(history: Dict[str, Dict[str, list]], save_dir: str):
    attack_types = ["Gaussian", "Laplace"]
    for attack in attack_types:
        plt.figure(figsize=(12, 8))
        
        for model_name, attack_history in history.items():
            accuracies = attack_history[attack]
            if not accuracies:
                continue
            epochs_logged = range(1, (len(accuracies) * ROBUSTNESS_LOG_INTERVAL) + 1, ROBUSTNESS_LOG_INTERVAL)
            plt.plot(epochs_logged, accuracies, label=model_name, marker='o', markersize=8)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy under Attack")
        plt.title(f"Model Robustness vs. Training Epochs ({attack} Attack)")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        save_path = os.path.join(save_dir, f"robustness_comparison_{attack.lower()}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"{attack} robustness plot saved to {save_path}")

# --- 5. Attack and Evaluation Functions ---
def gaussian_attack(data, epsilon):
    return data + torch.randn_like(data) * epsilon

def laplace_attack(data, epsilon):
    dist = torch.distributions.laplace.Laplace(0, epsilon / math.sqrt(2))
    return data + dist.sample(data.shape).to(data.device)
    
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
    X_train, y_train = create_dataset(n_samples=N_SAMPLES_TRAIN, noise=0.1, imbalance_ratio=0.8, random_state=SEED)
    X_test, y_test = create_dataset(n_samples=N_SAMPLES_TEST, noise=0.5, random_state=SEED+1)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    X_test_torch = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_torch = torch.from_numpy(y_test).long().to(DEVICE)

    # --- Models & History Logging---
    true_boundary_model = get_true_boundary_model(INPUT_DIM)
    
    LOG_EPOCH_INTERVAL = 6
    ROBUSTNESS_LOG_INTERVAL = 6
    
    model_names = ["SAA", "RGO", "LD", "Dual"]
    loss_histories = {name: [] for name in ["RGO", "LD", "Dual"]}
    robustness_history = {name: {"Gaussian": [], "Laplace": []} for name in model_names}
    
    # --- Pre-generate attacked test data ---
    X_test_gauss = gaussian_attack(X_test_torch, ATTACK_EPSILON)
    X_test_laplace = laplace_attack(X_test_torch, ATTACK_EPSILON)

    # --- SAA Training ---
    print("\n--- Training Standard (SAA) Model ---")
    saa_model = Classifier(INPUT_DIM).to(DEVICE)
    optimizer_saa = optim.Adam(saa_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(MAX_EPOCHS), desc="SAA Training"):
        epoch_loss = 0
        saa_model.train()
        for X_batch, y_batch in train_loader:
            optimizer_saa.zero_grad()
            loss = criterion(saa_model(X_batch.to(DEVICE)), y_batch.to(DEVICE))
            loss.backward()
            optimizer_saa.step()
            epoch_loss += loss.item()
            
        # if (epoch + 1) % LOG_EPOCH_INTERVAL == 0:
        #     loss_histories["SAA"].append(epoch_loss / len(train_loader))

        if (epoch + 1) % ROBUSTNESS_LOG_INTERVAL == 0:
            acc_gauss = evaluate_accuracy(saa_model, X_test_gauss, y_test_torch)
            acc_laplace = evaluate_accuracy(saa_model, X_test_laplace, y_test_torch)
            robustness_history["SAA"]["Gaussian"].append(acc_gauss)
            robustness_history["SAA"]["Laplace"].append(acc_laplace)

    trained_models = {"SAA": saa_model}
    
    # --- DRO Models Training ---
    dro_methods_to_train = ["RGO", "LD", "Dual"]
    
    for method in dro_methods_to_train:
        print(f"\n--- Training {method} Model ---")
        model = Classifier(INPUT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        gif_filenames = []

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
                
                elif method in ["RGO", "LD"]:
                    model.eval()
                    X_perturbed = rgo_sampler(X_batch, y_batch, model, epoch) if method == "RGO" else ld_sampler(X_batch, y_batch, model)
                    y_repeated = y_batch.repeat_interleave(NUM_SAMPLES_PER_POINT)
                    model.train()
                    loss = criterion(model(X_perturbed), y_repeated)

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % LOG_EPOCH_INTERVAL == 0:
                loss_histories[method].append(epoch_loss / len(train_loader))
            
            if epoch % ROBUSTNESS_LOG_INTERVAL == 0:
                acc_gauss = evaluate_accuracy(model, X_test_gauss, y_test_torch)
                acc_laplace = evaluate_accuracy(model, X_test_laplace, y_test_torch)
                robustness_history[method]["Gaussian"].append(acc_gauss)
                robustness_history[method]["Laplace"].append(acc_laplace)

            if method in ["RGO", "LD"]:
                model.eval()
                epoch_X_p = rgo_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model, epoch) if method == "RGO" else ld_sampler(train_dataset.tensors[0].to(DEVICE), train_dataset.tensors[1].to(DEVICE), model)
                frame_path = f"{OUTPUT_DIR}/{method}_frame_{epoch:03d}.png"
                plot_frame(model, true_boundary_model, X_train, y_train, epoch_X_p, f"Worst-case Distribution({method})", frame_path, method, epoch)
                gif_filenames.append(frame_path)

        trained_models[method] = model
        
        if imageio and method in ["RGO", "LD"]:
            print(f"Creating {method} evolution GIF...")
            gif_path = f"{OUTPUT_DIR}/{method}_two_moons_evolution.gif"
            with imageio.get_writer(gif_path, mode='I', duration=200, loop=0, format='gif') as writer:
                for filename in gif_filenames:
                    writer.append_data(imageio.imread(filename))
            print(f"GIF saved to {gif_path}")

    # --- Final Visualizations and Evaluations ---
    print("\n--- Generating Final Plots and Evaluations ---")
    
    # 1. Loss Comparison Plot
    plot_loss_comparison(loss_histories, f"{OUTPUT_DIR}/loss_comparison.png")
    print(f"Loss comparison plot saved to {OUTPUT_DIR}/loss_comparison.png")

    # 2. Boundary Comparison Plot
    visualize_all_boundaries(trained_models, true_boundary_model, X_train, y_train, "Final Decision Boundary Comparison", f"{OUTPUT_DIR}/final_boundary_comparison.png")
    print(f"Final boundary comparison plot saved to {OUTPUT_DIR}/final_boundary_comparison.png")
    
    # 3. Robustness Evolution Plot
    plot_robustness_comparison(robustness_history, OUTPUT_DIR)
    
    # 4. Final Robustness Evaluation
    print("\n--- Evaluating Final Model Robustness against Noise Attacks ---")
    print("-" * 70)
    print(f"Attack Strength (Epsilon): {ATTACK_EPSILON}")
    print("-" * 70)
    for name, model in trained_models.items():
        acc_clean = evaluate_accuracy(model, X_test_torch, y_test_torch)
        acc_gauss = evaluate_accuracy(model, X_test_gauss, y_test_torch)
        acc_laplace = evaluate_accuracy(model, X_test_laplace, y_test_torch)
        print(f"| Model: {name:<12} | Clean Acc: {acc_clean:7.2%} | Gaussian Acc: {acc_gauss:7.2%} | Laplace Acc: {acc_laplace:7.2%} |")
    print("-" * 70)
    
    print(f"\nExperiment complete. All results saved in '{OUTPUT_DIR}' directory.")