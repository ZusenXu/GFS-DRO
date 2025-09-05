import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D
from typing import Dict

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# 1. Data Generation (Consistent with Paper)
# ==============================================================================
def classification_SNVD20(num_samples=500, seed=42):
    """
    Generates synthetic data as described in Section 5.1 of the paper.
    """
    np.random.seed(seed)
    X = np.random.randn(num_samples * 5, 2)
    norms = np.linalg.norm(X, axis=1)
    y = np.sign(norms - np.sqrt(2))
    lower_bound = np.sqrt(2) / 1.3
    upper_bound = 1.3 * np.sqrt(2)
    mask = (norms < lower_bound) | (norms > upper_bound)
    X_filtered = X[mask][:num_samples]
    y_filtered = y[mask][:num_samples]
    return X_filtered, y_filtered

# ==============================================================================
# 2. Model Definition (Consistent with Paper)
# ==============================================================================
class SimpleNN(nn.Module):
    """
    A small neural network with 2 hidden layers of size 4 and 2, as per the paper.
    """
    def __init__(self, activation='elu'):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, 1)
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# ==============================================================================
# 3. Training Functions (Revised for Accuracy)
# ==============================================================================
def train_erm(model, X_train, y_train, epochs=300, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in tqdm(range(epochs), desc="Training ERM"):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, (y_train + 1) / 2) # Convert labels from {-1, 1} to {0, 1}
        loss.backward()
        optimizer.step()
    return model

def train_fgm(model, X_train, y_train, epsilon, epochs=300, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in tqdm(range(epochs), desc=f"Training FGM (ε={epsilon:.3f})"):
        X_adv = X_train.clone().detach().requires_grad_(True)
        outputs = model(X_adv)
        loss = criterion(outputs, (y_train + 1) / 2)
        loss.backward()
        with torch.no_grad():
            perturbed_data = X_train + epsilon * X_adv.grad.sign()
        optimizer.zero_grad()
        outputs_adv = model(perturbed_data)
        loss_adv = criterion(outputs_adv, (y_train + 1) / 2)
        loss_adv.backward()
        optimizer.step()
    return model

def train_wrm(model, X_train, y_train, gamma=2.0, epochs=300, lr=0.01, inner_lr=0.1, inner_steps=15):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in tqdm(range(epochs), desc="Training WRM (γ=2.0)"):
        x_adv = X_train.clone().detach().requires_grad_(True)
        # Inner loop: find worst-case perturbations via gradient ascent
        for _ in range(inner_steps):
            outputs_inner = model(x_adv)
            # Objective: max_z { l(θ;z) - γ * ||z - z_0||^2 }
            loss_inner = criterion(outputs_inner, (y_train + 1) / 2) - gamma * torch.mean(torch.sum((x_adv - X_train)**2, dim=1))
            grad = torch.autograd.grad(loss_inner, x_adv, retain_graph=True)[0]
            x_adv = x_adv + inner_lr * grad
        
        # Outer loop: update model parameters on the adversarial examples
        optimizer.zero_grad()
        loss_outer = criterion(model(x_adv.detach()), (y_train + 1) / 2)
        loss_outer.backward()
        optimizer.step()
    return model

def get_wrm_achieved_robustness(model, X_train, y_train, gamma=2.0, inner_lr=0.1, inner_steps=15):
    """
    Calculates the achieved robustness ρ_n for the trained WRM model, as per Eq. 23.
    """
    criterion = nn.BCEWithLogitsLoss()
    x_adv = X_train.clone().detach().requires_grad_(True)
    model.eval()
    for _ in range(inner_steps):
        outputs_inner = model(x_adv)
        loss_inner = criterion(outputs_inner, (y_train + 1) / 2) - gamma * torch.mean(torch.sum((x_adv - X_train)**2, dim=1))
        grad = torch.autograd.grad(loss_inner, x_adv, retain_graph=True)[0]
        x_adv = x_adv + inner_lr * grad
    
    with torch.no_grad():
        rho_n = torch.mean(torch.sum((x_adv - X_train)**2, dim=1))
    return rho_n.item()

# ==============================================================================
# 4. Visualization Function (Adapted from Your Example)
# ==============================================================================
def visualize_all_boundaries(models: Dict[str, nn.Module], X, y, title, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Adapt labels from {-1, 1} to {0, 1} for plotting consistency with your function
    y_plot = (y + 1) / 2

    # Plot data points
    ax.scatter(X[y_plot==1, 0], X[y_plot==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data (Y=1)', alpha=0.2)
    ax.scatter(X[y_plot==0, 0], X[y_plot==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data (Y=-1)', alpha=0.2)
    
    # Create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)

    colors = ['#FFD700', '#9932CC', '#32CD32'] # Gold, DarkOrchid, LimeGreen for ERM, FGM, WRM
    linestyles = ['--', '-.', '-']
    
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            # Use sigmoid for binary output and compare to 0.5
            Z = torch.sigmoid(model(grid)).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=[linestyles[i]], linewidths=2.5)

    # Create legend
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2.5, linestyle=linestyles[i], label=name) for i, name in enumerate(models.keys())]
    
    ax.legend(handles=legend_elements, loc='best', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_aspect('equal', 'box')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

# ==============================================================================
# 5. Main Execution Flow
# ==============================================================================
# Generate data
X_data, y_data = classification_SNVD20(num_samples=700, seed=42)
X_tensor = torch.tensor(X_data, dtype=torch.float32).to(DEVICE)
y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1).to(DEVICE)

# --- Train Models ---
print("--- Starting Model Training ---")

# Train ERM
erm_model = SimpleNN(activation='elu').to(DEVICE)
erm_model = train_erm(erm_model, X_tensor, y_tensor)

# Train WRM
wrm_model = SimpleNN(activation='elu').to(DEVICE)
wrm_model = train_wrm(wrm_model, X_tensor, y_tensor)

# Train FGM with epsilon derived from WRM
rho_n_wrm = get_wrm_achieved_robustness(wrm_model, X_tensor, y_tensor)
epsilon_for_fgm = np.sqrt(rho_n_wrm)
fgm_model = SimpleNN(activation='elu').to(DEVICE)
fgm_model = train_fgm(fgm_model, X_tensor, y_tensor, epsilon=epsilon_for_fgm)

print("--- Model Training Complete ---\n")

# --- Visualize Results ---
models_to_plot = {
    "ERM": erm_model,
    "FGM": fgm_model,
    "WRM": wrm_model
}

visualize_all_boundaries(
    models_to_plot,
    X_data,
    y_data,
    title="Decision Boundaries (ELU Activation)",
    save_path=r"./circle_results/decision_boundaries_replication.png"
)