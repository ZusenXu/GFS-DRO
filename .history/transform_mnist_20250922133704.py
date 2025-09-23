import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 定义您的函数和全局变量 (与之前相同) ---
NUM_SAMPLES_PER_POINT = 1
LAMBDA_PARAM = 0.1
EPSILON = 0.01
MAX_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inner_lr = 1e-2
inner_steps = 300

def wgf_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    """
    WGF 采样器: 使用 Langevin 动力学从最差情况分布中采样。
    (您的代码原文)
    """
    x_clone = x_original_batch.clone().detach().unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1, -1, -1).contiguous().view(-1, *x_original_batch.shape[1:])
    y_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1, -1, -1).reshape(-1, *x_original_batch.shape[1:])
    current_inner_steps = int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))
    model.eval()
    for _ in range(current_inner_steps):
        x_clone.requires_grad_(True)
        x_clone_on_device = x_clone.to(DEVICE)
        logits = model(x_clone_on_device)
        loss_values = nn.CrossEntropyLoss(reduction='none')(logits, y_repeated.to(DEVICE))
        if x_clone.grad is not None:
            x_clone.grad.zero_()
        grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values, device=DEVICE))
        grads = grads.to(x_clone.device)
        x_clone = x_clone.detach()
        mean = x_clone + lr * (grads - 2 * LAMBDA_PARAM * (x_clone - x_original_expanded))
        std_dev = torch.sqrt(torch.tensor(2 * lr * LAMBDA_PARAM * EPSILON, device=x_clone.device))
        noise = torch.randn_like(mean) * std_dev
        x_clone = mean + noise
        x_clone = torch.clamp(x_clone, 0, 1)
    return x_clone.detach()

# --- 2. 准备一个预训练的MNIST分类模型 (与之前相同) ---
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

def train_model():
    print("开始训练一个简单的MNIST分类器...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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

model = train_model()

# --- 3. (新) 搜索损失最高的“困难样本” ---
def find_hardest_sample_correctly_classified(model, loader, criterion):
    """
    遍历数据集，找到模型能够正确分类但损失函数值最高的样本。
    """
    print("\n正在搜索测试集中损失最高的（但仍分类正确）的样本...")
    max_loss = -1.0
    hardest_sample = None
    hardest_label = None
    original_image = None # 用于可视化的非归一化图像
    
    model.eval()
    with torch.no_grad():
        # 我们需要两个版本的loader，一个归一化用于模型输入，一个不归一化用于后续处理
        transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        transform_plain = transforms.ToTensor()
        
        dataset_norm = MNIST(root='./data', train=False, download=True, transform=transform_norm)
        dataset_plain = MNIST(root='./data', train=False, download=True, transform=transform_plain)
        
        loader_norm = DataLoader(dataset_norm, batch_size=1, shuffle=False)
        loader_plain = DataLoader(dataset_plain, batch_size=1, shuffle=False)

        # 同时遍历两个loader
        for (img_norm, lbl), (img_plain, _) in zip(loader_norm, loader_plain):
            img_norm, lbl = img_norm.to(DEVICE), lbl.to(DEVICE)
            
            output = model(img_norm)
            pred = torch.argmax(output, dim=1)
            
            # 检查是否分类正确
            if pred.item() == lbl.item():
                loss = criterion(output, lbl)
                if loss.item() > max_loss:
                    max_loss = loss.item()
                    hardest_sample = img_norm.cpu() # 存储归一化后的图像用于扰动
                    hardest_label = lbl.cpu()
                    original_image = img_plain.cpu() # 存储原始图像用于显示
                    
    print(f"搜索完成！找到的最难样本标签为 {hardest_label.item()}，其初始损失为: {max_loss:.4f}")
    return original_image, hardest_sample, hardest_label

# 定义损失函数，用于搜索
criterion = nn.CrossEntropyLoss()
# 运行搜索函数
x_original_unnormalized, x_normalized, y_original = find_hardest_sample_correctly_classified(model, test_loader, criterion)


# --- 4. 运行扰动函数并可视化结果 (与之前类似) ---
print("\n正在对找到的'最难样本'应用WGF采样器...")
perturbed_normalized = wgf_sampler(x_normalized, y_original, model, epoch=10)

# 为了可视化，我们需要将归一化的图像转换回[0,1]范围
unorm = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
perturbed_image = unorm(perturbed_normalized)
perturbed_image = torch.clamp(perturbed_image, 0, 1)

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

axes[0].imshow(x_original_unnormalized.squeeze(), cmap='gray')
axes[0].set_title(f'Hardest Original Image\nLabel: {y_original.item()}\nModel Prediction: {original_pred.item()}')
axes[0].axis('off')

axes[1].imshow(perturbed_image.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title(f'Perturbed Image\nModel Prediction: {perturbed_pred.item()}')
axes[1].axis('off')

perturbation = (perturbed_image - x_original_unnormalized).abs()
axes[2].imshow(perturbation.squeeze().cpu().numpy(), cmap='gray')
axes[2].set_title('The Perturbation (Noise)')
axes[2].axis('off')

plt.show()