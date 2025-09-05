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
MAX_EPOCHS = 250 # 增加训练轮数以确保收敛
BATCH_SIZE = 128
LR = 1e-2
SEED = 42

# DRO (multiLD) 训练参数 (来自 two_moon.py)
EPSILON = 0.05 # 稍微增大了扰动，以便在环形数据上看到更明显的效果
LAMBDA_PARAM = 10.0
NUM_SAMPLES_PER_POINT = 8


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

# --- 3. 模型定义 (根据您的要求新建) ---
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

# --- 4. 训练逻辑 (来自 two_moon.py) ---
def multiLD(model, optimizer, train_loader, loss_fn, lambda_param, epsilon, num_samples):
    """
    来自您提供的 two_moon.py 文件中的 multi-level DRO 训练逻辑。
    """
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # 核心步骤：为每个数据点生成对抗性/扰动样本
        x_tilde = x.unsqueeze(1).repeat(1, num_samples, 1)
        x_tilde += epsilon * torch.randn_like(x_tilde) # 添加高斯噪声
        x_tilde = x_tilde.view(-1, x.shape[1])

        y_pred = model(x_tilde)
        # 计算每个扰动样本的损失
        losses = loss_fn(y_pred, y.repeat_interleave(num_samples))
        losses = losses.view(-1, num_samples)

        # Sinkhorn-like 步骤：找到最坏情况下的概率分布 p
        p = -losses.detach() / lambda_param # 使用 .detach() 避免双重反向传播
        p = torch.softmax(p, dim=1)

        # 用最坏情况的分布 p 来加权损失
        loss = torch.mean(torch.sum(p * losses, dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

# --- 5. 可视化函数 (适配自您的模板) ---
def visualize_all_boundaries(models: Dict[str, nn.Module], X, y, title, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制数据点
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', s=25, label='Class 1', alpha=0.3)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', s=25, label='Class 0', alpha=0.3)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)

    colors = ['#FF4500', '#1E90FF'] # 橙红色, 道奇蓝
    linestyles = ['-', '--']
    
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=[linestyles[i]], linewidths=3)

    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=3, linestyle=linestyles[0], label="DRO (ReLU)"),
        Line2D([0], [0], color=colors[1], lw=3, linestyle=linestyles[1], label="DRO (ELU)")
    ]
    
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
    # 1. 使用论文5.1节的数据生成器
    X_train, y_train = classification_SNVD20(N_SAMPLES_TRAIN, seed=SEED)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # CrossEntropyLoss 需要 reduction='none' 以便 multiLD 算法可以处理每个样本的损失
    criterion = nn.CrossEntropyLoss(reduction='none')

    # --- 2. 训练 ReLU 模型 ---
    print("--- Training DRO Model with ReLU Activation using multiLD ---")
    model_relu = TwoLayerNet(activation='relu').to(DEVICE)
    optimizer_relu = optim.Adam(model_relu.parameters(), lr=LR) # Adam可能比SGD收敛更快
    
    for epoch in range(MAX_EPOCHS):
        loss = multiLD(model_relu, optimizer_relu, train_loader, criterion, LAMBDA_PARAM, EPSILON, NUM_SAMPLES_PER_POINT)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Loss: {loss:.4f}")

    # --- 3. 训练 ELU 模型 ---
    print("\n--- Training DRO Model with ELU Activation using multiLD ---")
    model_elu = TwoLayerNet(activation='elu').to(DEVICE)
    optimizer_elu = optim.Adam(model_elu.parameters(), lr=LR)

    for epoch in range(MAX_EPOCHS):
        loss = multiLD(model_elu, optimizer_elu, train_loader, criterion, LAMBDA_PARAM, EPSILON, NUM_SAMPLES_PER_POINT)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Loss: {loss:.4f}")
            
    print("\n--- Model Training Complete ---")

    # --- 4. 可视化结果 ---
    print("\nPlotting decision boundaries...")
    trained_models = {
        "DRO (ReLU)": model_relu,
        "DRO (ELU)": model_elu
    }
    
    visualize_all_boundaries(
        trained_models,
        X_train,
        y_train,
        title="Decision Boundaries using multiLD Training",
        save_path="dro_article_data_multild_training.png"
    )
