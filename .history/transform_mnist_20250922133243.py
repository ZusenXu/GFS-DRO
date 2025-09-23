import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 定义您的函数和缺失的全局变量 ---
# (这些是基于函数上下文推断的合理默认值)
NUM_SAMPLES_PER_POINT = 1
LAMBDA_PARAM = 0.1  # 正则化强度，控制扰动大小
EPSILON = 0.1      # 噪声项的尺度
MAX_EPOCHS = 10     # 用于计算内部步数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inner_lr = 1e-3     # 内部Langevin动力学的学习率
inner_steps = 300    # 内部Langevin动力学的步数

def wgf_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    """
    WGF 采样器: 使用 Langevin 动力学从最差情况分布中采样。
    WGF Sampler: Uses Langevin dynamics to sample from the worst-case distribution.
    (您的代码原文)
    """
    # 初始化多个粒子 (Initialize multiple particles)
    x_clone = x_original_batch.clone().detach().unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1, -1, -1).contiguous().view(-1, *x_original_batch.shape[1:])
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1, -1, -1).reshape(-1, *x_original_batch.shape[1:])

    # 调整内部迭代次数
    current_inner_steps = int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))
    
    model.eval() # 将模型设置为评估模式

    for _ in range(current_inner_steps):
        x_clone.requires_grad_(True)
        # 将x_clone送入模型前确保它在正确的设备上
        x_clone_on_device = x_clone.to(DEVICE)
        
        logits = model(x_clone_on_device)
        loss_values = nn.CrossEntropyLoss(reduction='none')(logits, y_repeated.to(DEVICE))
        
        # 清除旧梯度
        if x_clone.grad is not None:
            x_clone.grad.zero_()
            
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values, device=DEVICE))
        
        # 确保梯度回到CPU上进行后续计算
        grads = grads.to(x_clone.device)
        x_clone = x_clone.detach()

        # Langevin 动力学更新 (Langevin dynamics update)
        mean = x_clone + lr * (grads - 2 * LAMBDA_PARAM * (x_clone - x_original_expanded))
        std_dev = torch.sqrt(torch.tensor(2 * lr * LAMBDA_PARAM * EPSILON, device=x_clone.device))
        noise = torch.randn_like(mean) * std_dev
        x_clone = mean + noise
        
        # 保持像素值在合理范围内 (可选，但推荐)
        x_clone = torch.clamp(x_clone, 0, 1)

    return x_clone.detach()

# --- 2. 准备一个预训练的MNIST分类模型 ---

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

# 训练一个模型 (为了演示，我们只训练几轮)
def train_model():
    print("开始训练一个简单的MNIST分类器...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练2个epoch就足够用于演示了
    for epoch in range(2):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/2], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    print("模型训练完成。")
    return model

# 训练或加载模型
# 注意：为了可复现性，每次运行此脚本都会重新训练一个简单的模型。
# 在实际应用中，您应该保存并加载一个训练好的模型。
model = train_model()

# --- 3. 加载MNIST数据并选择一个样本 ---

# 使用不同的变换，因为我们的扰动是在原始像素空间[0,1]上进行的
transform_test = transforms.ToTensor()
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 获取一张图片
x_original, y_original = next(iter(test_loader))
x_original, y_original = x_original.cpu(), y_original.cpu() # 确保在CPU上

# --- 4. 运行扰动函数并可视化结果 ---

print("\n正在对一张图片应用WGF采样器生成对抗性扰动...")
# 我们需要对输入进行归一化，以匹配模型的训练数据
transform_norm = transforms.Normalize((0.1307,), (0.3081,))
x_normalized = transform_norm(x_original)

# 调用函数
# 注意：这里我们假设epoch=5，这会影响内部迭代步数
perturbed_normalized = wgf_sampler(x_normalized, y_original, model, epoch=10)

# 为了可视化，我们需要将归一化的图像转换回[0,1]范围
# 这是归一化的逆操作
unorm = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
perturbed_image = unorm(perturbed_normalized)
perturbed_image = torch.clamp(perturbed_image, 0, 1) # 确保值在[0,1]

# --- 5. 展示结果 ---
model.to(DEVICE).eval()
with torch.no_grad():
    original_output = model(x_normalized.to(DEVICE))
    perturbed_output = model(perturbed_normalized.to(DEVICE))

original_pred = torch.argmax(original_output, dim=1)
perturbed_pred = torch.argmax(perturbed_output, dim=1)

print(f"\n原始标签: {y_original.item()}")
print(f"模型对原始图片的预测: {original_pred.item()}")
print(f"模型对扰动图片的预测: {perturbed_pred.item()}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(x_original.squeeze(), cmap='gray')
axes[0].set_title(f'Original Image\nLabel: {y_original.item()}\nModel Prediction: {original_pred.item()}')
axes[0].axis('off')

axes[1].imshow(perturbed_image.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title(f'Perturbed Image\nModel Prediction: {perturbed_pred.item()}')
axes[1].axis('off')

# 可视化扰动本身
perturbation = (perturbed_image - x_original).abs()
axes[2].imshow(perturbation.squeeze().cpu().numpy(), cmap='gray')
axes[2].set_title('The Perturbation (Noise)')
axes[2].axis('off')

plt.savefig("wgf_mnist_perturbation.png")
plt.show()