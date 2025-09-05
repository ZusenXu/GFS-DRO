import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# 1. 数据生成函数 (与之前相同)
# ==============================================================================
def classification_SNVD20(num_samples=500, seed=42):
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
# 2. 模型定义 (与之前相同)
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
# 3. 训练函数 (与之前相同)
# ==============================================================================
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

def train_fgm(model, X_train, y_train, epsilon=0.1, epochs=200, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in tqdm(range(epochs), desc="Training FGM"):
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

def train_wrm(model, X_train, y_train, gamma=2.0, epochs=200, lr=0.01, inner_lr=0.1, inner_steps=15):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in tqdm(range(epochs), desc="Training WRM"):
        x_adv = X_train.clone().detach().requires_grad_(True)
        for _ in range(inner_steps):
            loss_inner = criterion(model(x_adv), (y_train + 1) / 2) - gamma * torch.sum((x_adv - X_train)**2, dim=1).mean()
            grad = torch.autograd.grad(loss_inner, x_adv, create_graph=True)[0] # create_graph is needed for robust Hessians if any
            x_adv = x_adv + inner_lr * grad
        optimizer.zero_grad()
        loss_outer = criterion(model(x_adv.detach()), (y_train + 1) / 2)
        loss_outer.backward()
        optimizer.step()
    return model

# ==============================================================================
# 4. 修正后的可视化函数
# ==============================================================================
def plot_decision_boundary_corrected(models, X, y, titles):
    """
    修正后的决策边界绘制函数，风格与论文更接近。
    """
    plt.style.use('default') # 使用默认样式以获得白色背景
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制数据点
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.1, label='Class 1', s=10)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', alpha=1, label='Class -1', s=15, marker='o', facecolors='blue', edgecolors='blue')


    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    colors = ['#FFD700', '#9932CC', '#32CD32']  # Gold, DarkOrchid, LimeGreen (类似论文的黄、紫、绿)
    linewidths = [2.5, 2.5, 2.5]

    for i, (model, title) in enumerate(zip(models, titles)):
        model.eval()
        with torch.no_grad():
            Z = model(grid_tensor)
        Z = torch.sigmoid(Z).numpy().reshape(xx.shape)
        
        # 只绘制 0.5 的决策边界
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linewidths=linewidths[i], label=title)
    
    # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[0], lw=4, label=titles[0]),
                       Line2D([0], [0], color=colors[1], lw=4, label=titles[1]),
                       Line2D([0], [0], color=colors[2], lw=4, label=titles[2])]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax.set_title("Decision Boundaries (ELU Model)", fontsize=16)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(r"./circle_results/decision_boundaries_corrected.png", dpi=300)
    plt.show()


# ==============================================================================
# 5. 主执行流程
# ==============================================================================
# 生成数据
print("Generating synthetic data...")
X_data, y_data = classification_SNVD20(num_samples=500, seed=42)

# 初始化模型
activation_func = 'elu'
erm_model = SimpleNN(activation=activation_func)
fgm_model = SimpleNN(activation=activation_func)
wrm_model = SimpleNN(activation=activation_func)

# 训练模型
print("\n--- Starting Model Training ---")
erm_model = train_erm(erm_model, torch.tensor(X_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32).view(-1, 1))
fgm_model = train_fgm(fgm_model, torch.tensor(X_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32).view(-1, 1), epsilon=0.2)
wrm_model = train_wrm(wrm_model, torch.tensor(X_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32).view(-1, 1), gamma=2.0)
print("--- Model Training Complete ---\n")

# 可视化结果
print("Plotting decision boundaries...")
plot_decision_boundary_corrected(
    [erm_model, fgm_model, wrm_model],
    X_data,
    y_data,
    ["ERM", "FGM", "WRM"]
)