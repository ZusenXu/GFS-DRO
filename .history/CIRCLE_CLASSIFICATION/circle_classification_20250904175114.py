import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ==============================================================================
# 1. 数据生成和可视化函数 (来自您提供的文件)
# ==============================================================================

def draw_classification(X, y, save_dir=None):
    """
    绘制分类数据的散点图。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==-1][:, 0], X[y==-1][:, 1], c='royalblue', label='Class -1', alpha=0.7)
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='darkorange', label='Class 1', alpha=0.7)
    plt.title('Synthetic Data for Classification')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_dir:
        plt.savefig(save_dir)
        print(f"Data visualization saved to {save_dir}")
    else:
        plt.show()

def classification_SNVD20(num_samples=500, seed=42, visualize=False, save_dir=None):
    """
    遵循“Certifying Some Distributional Robustness with Principled Adversarial Training”论文中5.1节的思想
    链接: https://arxiv.org/pdf/1710.10571
    """
    np.random.seed(seed)
    # 增加样本量以便过滤后仍有足够数据
    X = np.random.randn(num_samples * 5, 2) 
    norms = np.linalg.norm(X, axis=1)
    y = np.sign(norms - np.sqrt(2))
    lower_bound = np.sqrt(2) / 1.3
    upper_bound = 1.3 * np.sqrt(2)
    mask = (norms < lower_bound) | (norms > upper_bound)

    X_filtered = X[mask][:num_samples]
    y_filtered = y[mask][:num_samples]

    if visualize:
        draw_classification(X_filtered, y_filtered, save_dir)

    return X_filtered, y_filtered


def plot_decision_boundary(models, X, y, titles):
    """
    为多个模型绘制决策边界。
    """
    plt.figure(figsize=(20, 5))
    
    for i, (model, title) in enumerate(zip(models, titles)):
        plt.subplot(1, len(models), i + 1)
        # 绘制数据点
        plt.scatter(X[y==-1][:, 0], X[y==-1][:, 1], c='royalblue', label='Class -1', alpha=0.3)
        plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='darkorange', label='Class 1', alpha=0.3)

        # 创建一个网格来绘制决策边界
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # 获取模型在网格点上的预测
        grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        model.eval() # 设置为评估模式
        with torch.no_grad():
            Z = model(grid_tensor)
        Z = torch.sigmoid(Z).numpy().reshape(xx.shape)
        
        # 绘制等高线，即决策边界
        contour_colors = ['yellow', 'purple', 'green']
        plt.contour(xx, yy, Z, levels=[0.5], colors=[contour_colors[i]], linewidths=3)
        
        plt.title(title, fontsize=14)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Decision Boundaries Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(r"./circle_results/decision_boundaries_comparison.png")
    plt.show()

# ==============================================================================
# 2. 模型定义
# ==============================================================================

class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim1=4, hidden_dim2=2, output_dim=1, activation='elu'):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# ==============================================================================
# 3. 训练函数
# ==============================================================================

# ERM 训练
def train_erm(model, X_train, y_train, epochs=200, lr=0.01):
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

# FGM 训练
def train_fgm(model, X_train, y_train, epsilon=0.1, epochs=200, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in tqdm(range(epochs), desc="Training FGM"):
        # 允许计算输入数据的梯度
        X_adv = X_train.clone().detach().requires_grad_(True)
        
        outputs = model(X_adv)
        loss = criterion(outputs, (y_train + 1) / 2)
        loss.backward()

        # 生成对抗样本
        with torch.no_grad():
            perturbed_data = X_train + epsilon * X_adv.grad.sign()
        
        # 在对抗样本上重新计算损失并更新模型
        optimizer.zero_grad()
        outputs_adv = model(perturbed_data)
        loss_adv = criterion(outputs_adv, (y_train + 1) / 2)
        loss_adv.backward()
        optimizer.step()
    return model

# WRM 训练 (手动实现核心逻辑)
def train_wrm(model, X_train, y_train, gamma=2.0, epochs=200, lr=0.01, 
              inner_lr=0.1, inner_steps=15):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    
    for _ in tqdm(range(epochs), desc="Training WRM"):
        # 1. 寻找最差情况的扰动 (inner loop)
        x_adv = X_train.clone().detach().requires_grad_(True)
        
        for _ in range(inner_steps):
            # l(theta; z) - gamma * c(z, z_0)
            # c(z, z_0) is ||z - z_0||_2^2
            loss_inner = criterion(model(x_adv), (y_train + 1) / 2) - gamma * torch.sum((x_adv - X_train)**2, dim=1).mean()
            
            # 我们要最大化这个内部损失，所以执行梯度上升
            grad = torch.autograd.grad(loss_inner, x_adv)[0]
            x_adv.data = x_adv.data + inner_lr * grad.data
            
        # 2. 更新模型参数 (outer loop)
        optimizer.zero_grad()
        # 计算在对抗样本上的损失
        loss_outer = criterion(model(x_adv.detach()), (y_train + 1) / 2)
        loss_outer.backward()
        optimizer.step()
        
    return model

# ==============================================================================
# 4. 主执行流程
# ==============================================================================

# 生成数据
print("Generating synthetic data...")
X_data, y_data = classification_SNVD20(num_samples=500, seed=42, visualize=True)
X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

# 初始化三个模型 (使用ELU激活函数，与论文图1b一致)
activation_func = 'elu'
erm_model = SimpleNN(activation=activation_func)
fgm_model = SimpleNN(activation=activation_func)
wrm_model = SimpleNN(activation=activation_func)

# 训练模型
print("\n--- Starting Model Training ---")
erm_model = train_erm(erm_model, X_tensor, y_tensor)
fgm_model = train_fgm(fgm_model, X_tensor, y_tensor, epsilon=0.2) # Epsilon需要调整以获得可比较的结果
wrm_model = train_wrm(wrm_model, X_tensor, y_tensor, gamma=2.0) # Gamma=2.0如论文所述
print("--- Model Training Complete ---\n")

# 可视化结果
print("Plotting decision boundaries...")
plot_decision_boundary(
    [erm_model, fgm_model, wrm_model],
    X_data,
    y_data,
    ["ERM (ELU)", "FGM (ELU)", "WRM (ELU)"]
)