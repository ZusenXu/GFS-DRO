import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Dict

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- 1. 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 实验参数
N_SAMPLES_TRAIN = 700
MAX_EPOCHS = 300 # 增加训练轮数以确保收敛
BATCH_SIZE = 128
LR = 1e-2
SEED = 42

# DRO (multiLD) 训练参数
EPSILON_MULTILD = 0.05
LAMBDA_PARAM = 10.0
NUM_SAMPLES_PER_POINT = 8

# Sinha WRM 训练参数
GAMMA_SINHA = 2.0
INNER_LR_SINHA = 0.1
INNER_STEPS_SINHA = 15


# --- 2. 数据采样器 (来自论文 5.1) ---
def classification_SNVD20(n_samples, seed=42):
    """
    生成论文 Section 5.1 中描述的合成数据。
    """
    np.random.seed(seed)
    # 生成更多点以确保过滤后有足够样本
    X = np.random.randn(n_samples * 5, 2)
    norms = np.linalg.norm(X, axis=1)
    y = np.sign(norms - np.sqrt(2))
    lower_bound = np.sqrt(2) / 1.3
    upper_bound = 1.3 * np.sqrt(2)
    mask = (norms < lower_bound) | (norms > upper_bound)
    X_filtered = X[mask][:n_samples]
    y_filtered = y[mask][:n_samples]
    # 将标签从 {-1, 1} 转换为 {0, 1}
    y_final = ((y_filtered + 1) / 2).astype(int)
    return X_filtered, y_final

# --- 3. 模型定义 ---
class TwoLayerNet(nn.Module):
    """
    一个两层的神经网络，包含一个4单元的隐藏层和一个2单元的输出层。
    """
    def __init__(self, activation='relu'):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Activation '{activation}' not supported.")
        self.fc2 = nn.Linear(4, 2) # 2个输出单元对应2个类别

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 4. 训练逻辑 ---
def multiLD(model, optimizer, train_loader, loss_fn, lambda_param, epsilon, num_samples):
    """
    来自您提供的 two_moon.py 文件中的 multi-level DRO 训练逻辑。
    """
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        x_tilde = x.unsqueeze(1).repeat(1, num_samples, 1)
        x_tilde += epsilon * torch.randn_like(x_tilde)
        x_tilde = x_tilde.view(-1, x.shape[1])

        y_pred = model(x_tilde)
        losses = loss_fn(y_pred, y.repeat_interleave(num_samples))
        losses = losses.view(-1, num_samples)

        p = -losses.detach() / lambda_param
        p = torch.softmax(p, dim=1)

        loss = torch.mean(torch.sum(p * losses, dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def train_sinha_wrm(model, optimizer, train_loader, loss_fn, gamma, inner_lr, inner_steps):
    """
    来自论文原文的 principled adversarial training (WRM) 训练逻辑。
    """
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # 内部循环: 通过梯度上升寻找最差扰动
        x_adv = x.clone().detach().requires_grad_(True)
        for _ in range(inner_steps):
            outputs_inner = model(x_adv)
            # 目标: max_z { l(θ;z) - γ * ||z - z_0||^2 }
            # 注意：loss_fn返回的是每个样本的损失
            individual_losses = loss_fn(outputs_inner, y)
            cost = gamma * torch.sum((x_adv - x)**2, dim=1)
            objective = torch.mean(individual_losses - cost)
            
            # 使用autograd计算梯度并执行梯度上升
            grad = torch.autograd.grad(objective, x_adv, create_graph=True)[0]
            x_adv = x_adv + inner_lr * grad
        
        # 外部循环: 在对抗样本上更新模型参数
        optimizer.zero_grad()
        # 移除 .detach()，使得梯度可以流回模型参数
        loss_outer = torch.mean(loss_fn(model(x_adv), y))
        loss_outer.backward()
        optimizer.step()
        total_loss += loss_outer.item()

    return total_loss / len(train_loader)


# --- 5. 可视化函数 ---
def visualize_all_boundaries(models: Dict[str, nn.Module], X, y, title, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', s=25, label='Class 1', alpha=0.3)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', s=25, label='Class 0', alpha=0.3)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)

    colors = ['#FF4500', '#1E90FF', '#FFD700', '#32CD32'] # Red, Blue, Gold, Green
    linestyles = ['-', '--', '-', '--']
    
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=[linestyles[i]], linewidths=3)

    legend_elements = [Line2D([0], [0], color=colors[i], lw=3, linestyle=linestyles[i], label=name) for i, name in enumerate(models.keys())]
    
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

# --- 6. 主执行流程 ---
if __name__ == "__main__":
    X_train, y_train = classification_SNVD20(N_SAMPLES_TRAIN, seed=SEED)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # --- 训练 multiLD 模型 ---
    print("--- Training multiLD Models ---")
    model_multild_relu = TwoLayerNet(activation='relu').to(DEVICE)
    optimizer_multild_relu = optim.Adam(model_multild_relu.parameters(), lr=LR)
    for epoch in tqdm(range(MAX_EPOCHS), desc="multiLD (ReLU)"):
        multiLD(model_multild_relu, optimizer_multild_relu, train_loader, criterion, LAMBDA_PARAM, EPSILON_MULTILD, NUM_SAMPLES_PER_POINT)
        
    model_multild_elu = TwoLayerNet(activation='elu').to(DEVICE)
    optimizer_multild_elu = optim.Adam(model_multild_elu.parameters(), lr=LR)
    for epoch in tqdm(range(MAX_EPOCHS), desc="multiLD (ELU)"):
        multiLD(model_multild_elu, optimizer_multild_elu, train_loader, criterion, LAMBDA_PARAM, EPSILON_MULTILD, NUM_SAMPLES_PER_POINT)

    # --- 训练 Sinha-WRM 模型 ---
    print("\n--- Training Sinha-WRM Models ---")
    # 注意：Sinha-WRM的损失函数需要每个样本的损失，所以我们使用reduction='none'，并在外部取mean
    criterion_sinha = nn.CrossEntropyLoss(reduction='none') 
    model_sinha_relu = TwoLayerNet(activation='relu').to(DEVICE)
    optimizer_sinha_relu = optim.SGD(model_sinha_relu.parameters(), lr=LR)
    for epoch in tqdm(range(MAX_EPOCHS), desc="Sinha-WRM (ReLU)"):
        train_sinha_wrm(model_sinha_relu, optimizer_sinha_relu, train_loader, criterion_sinha, GAMMA_SINHA, INNER_LR_SINHA, INNER_STEPS_SINHA)

    model_sinha_elu = TwoLayerNet(activation='elu').to(DEVICE)
    optimizer_sinha_elu = optim.SGD(model_sinha_elu.parameters(), lr=LR)
    for epoch in tqdm(range(MAX_EPOCHS), desc="Sinha-WRM (ELU)"):
        train_sinha_wrm(model_sinha_elu, optimizer_sinha_elu, train_loader, criterion_sinha, GAMMA_SINHA, INNER_LR_SINHA, INNER_STEPS_SINHA)
            
    print("\n--- Model Training Complete ---")

    # --- 可视化结果 ---
    print("\nPlotting decision boundaries...")
    trained_models = {
        "multiLD (ReLU)": model_multild_relu,
        "multiLD (ELU)": model_multild_elu,
        "Sinha-WRM (ReLU)": model_sinha_relu,
        "Sinha-WRM (ELU)": model_sinha_elu
    }
    
    visualize_all_boundaries(
        trained_models,
        X_train,
        y_train,
        title="Decision Boundary Comparison: multiLD vs. Sinha-WRM",
        save_path="dro_training_comparison.png"
    )

