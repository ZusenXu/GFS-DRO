import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D
from typing import Dict

# 确定运行设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# 1. 数据生成函数 (与原文一致)
# ==============================================================================
def classification_SNVD20(num_samples=700, seed=42):
    """
    生成论文 Section 5.1 中描述的合成数据。
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
# 2. 模型定义 (与原文一致)
# ==============================================================================
class SimpleNN(nn.Module):
    """
    实现论文中描述的具有2个隐藏层的神经网络。
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
# 3. 训练函数 (已修正并严格对齐原文)
# ==============================================================================
def train_erm(model, X_train, y_train, epochs=400, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in tqdm(range(epochs), desc="Training ERM"):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, (y_train + 1) / 2)
        loss.backward()
        optimizer.step()
    return model

def train_fgm(model, X_train, y_train, epsilon, epochs=400, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    pbar = tqdm(range(epochs), desc=f"Training FGM (ε={epsilon:.4f})")
    for _ in pbar:
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
        pbar.set_postfix({"loss": loss_adv.item()})
    return model

def train_wrm(model, X_train, y_train, gamma=2.0, epochs=400, lr=0.01, inner_lr=0.1, inner_steps=15):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    pbar = tqdm(range(epochs), desc=f"Training WRM (γ={gamma})")
    for _ in pbar:
        x_adv = X_train.clone().detach().requires_grad_(True)
        # 内部循环: 通过梯度上升寻找最差扰动
        for _ in range(inner_steps):
            outputs_inner = model(x_adv)
            loss_inner = criterion(outputs_inner, (y_train + 1) / 2) - gamma * torch.mean(torch.sum((x_adv - X_train)**2, dim=1))
            # 关键修正：使用create_graph=True来构建可反向传播的梯度图
            grad = torch.autograd.grad(-loss_inner, x_adv, create_graph=True)[0] # 最小化负loss等价于最大化原loss
            x_adv = x_adv - inner_lr * grad

        # 外部循环: 在对抗样本上更新模型参数
        optimizer.zero_grad()
        # 关键修正：移除 .detach()，使得梯度可以从对抗样本流回模型参数
        loss_outer = criterion(model(x_adv), (y_train + 1) / 2)
        loss_outer.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss_outer.item()})
    return model

def get_wrm_achieved_robustness(model, X_train, y_train, gamma=2.0, inner_lr=0.1, inner_steps=15):
    """
    根据 Eq. 23 计算训练好的WRM模型所实现的鲁棒性 ρ_n。
    """
    criterion = nn.BCEWithLogitsLoss()
    x_adv = X_train.clone().detach().requires_grad_(True)
    model.eval()
    for _ in range(inner_steps):
        outputs_inner = model(x_adv)
        loss_inner = criterion(outputs_inner, (y_train + 1) / 2) - gamma * torch.mean(torch.sum((x_adv - X_train)**2, dim=1))
        grad = torch.autograd.grad(-loss_inner, x_adv)[0]
        x_adv = x_adv - inner_lr * grad
    with torch.no_grad():
        rho_n = torch.mean(torch.sum((x_adv - X_train)**2, dim=1))
    return rho_n.item()

# ==============================================================================
# 4. 可视化函数 (根据您的模板适配)
# ==============================================================================
def visualize_all_boundaries(models: Dict[str, nn.Module], X, y, title, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    y_plot = (y + 1) / 2

    ax.scatter(X[y_plot==1, 0], X[y_plot==1, 1], c='darkorange', marker='o', s=20, label='Class 1', alpha=0.3)
    ax.scatter(X[y_plot==0, 0], X[y_plot==0, 1], c='dodgerblue', marker='o', s=20, label='Class -1', alpha=0.3)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)

    colors = ['gold', 'darkviolet', 'limegreen']
    linestyles = ['--', '-.', '-']
    
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            Z = torch.sigmoid(model(grid)).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=[linestyles[i]], linewidths=2.5)

    legend_elements = [Line2D([0], [0], color=colors[i], lw=2.5, linestyle=linestyles[i], label=name) for i, name in enumerate(models.keys())]
    
    ax.legend(handles=legend_elements, loc='best', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_aspect('equal', 'box')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()

# ==============================================================================
# 5. 主执行流程
# ==============================================================================
# 生成数据
X_data, y_data = classification_SNVD20(num_samples=700, seed=42)
X_tensor = torch.tensor(X_data, dtype=torch.float32).to(DEVICE)
y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1).to(DEVICE)

# --- 训练模型 (严格遵循原文流程) ---
print("--- Starting Model Training (now with corrected logic) ---")

# 1. 训练 ERM
erm_model = SimpleNN(activation='elu').to(DEVICE)
train_erm(erm_model, X_tensor, y_tensor)

# 2. 训练 WRM
wrm_model = SimpleNN(activation='elu').to(DEVICE)
train_wrm(wrm_model, X_tensor, y_tensor)

# 3. 根据 WRM 结果训练 FGM
rho_n_wrm = get_wrm_achieved_robustness(wrm_model, X_tensor, y_tensor)
epsilon_for_fgm = np.sqrt(rho_n_wrm)
fgm_model = SimpleNN(activation='elu').to(DEVICE)
train_fgm(fgm_model, X_tensor, y_tensor, epsilon=epsilon_for_fgm)

print("--- Model Training Complete ---\n")

# --- 可视化最终结果 ---
models_to_plot = {
    "ERM": erm_model,
    "FGM": fgm_model,
    "WRM": wrm_model
}
visualize_all_boundaries(
    models_to_plot,
    X_data,
    y_data,
    title="Decision Boundaries (ELU Activation, Corrected Training)",
    save_path=r"./circle_results/decision_boundaries_replication_corrected.png"
)