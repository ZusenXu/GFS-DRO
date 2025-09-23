import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 参数设置 (保持不变) ---
NUM_SAMPLES_PER_POINT = 1
LAMBDA_PARAM = 0.1
EPSILON = 0.01
MAX_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inner_lr = 1e-2
inner_steps = 300

# --- 2. 目标性扰动函数 (保持不变) ---
def wgf_sampler_targeted(x_original_batch, y_original_batch, y_target_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    """
    目标性WGF采样器: 不仅远离原始类别，而且靠近目标类别。
    """
    x_clone = x_original_batch.clone().detach().unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1, -1, -1).contiguous().view(-1, *x_original_batch.shape[1:])
    y_original_repeated = y_original_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    y_target_repeated = y_target_batch.repeat_interleave(NUM_SAMPLES_PER_POINT, dim=0)
    x_original_expanded = x_original_batch.unsqueeze(1).expand(-1, NUM_SAMPLES_PER_POINT, -1, -1, -1).reshape(-1, *x_original_batch.shape[1:])
    current_inner_steps = int(max(5, inner_steps * (epoch + 1) / MAX_EPOCHS))
    print(f"Executing targeted perturbation with {current_inner_steps} steps and lr={lr:.4f}...")
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    for i in range(current_inner_steps):
        x_clone.requires_grad_(True)
        x_clone_on_device = x_clone.to(DEVICE)
        logits = model(x_clone_on_device)
        loss_away_from_original = criterion(logits, y_original_repeated.to(DEVICE))
        loss_towards_target = criterion(logits, y_target_repeated.to(DEVICE))
        loss_values = loss_away_from_original - loss_towards_target
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

# --- 3. MNIST分类模型定义与训练 (保持不变) ---
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
    print("Starting to train a simple MNIST classifier...")
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
    print("Model training complete.")
    return model

# --- 4. (已修改) 按指定类别搜索“最难样本”的函数 ---
def find_hardest_sample_for_class(model, criterion, class_to_find):
    print(f"\nSearching for the highest-loss sample for class '{class_to_find}'...")
    max_loss = -1.0
    hardest_sample_normalized = None
    hardest_label = None
    original_image_unnormalized = None 
    
    model.eval()
    with torch.no_grad():
        transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        transform_plain = transforms.ToTensor()
        dataset_norm = MNIST(root='./data', train=False, download=True, transform=transform_norm)
        dataset_plain = MNIST(root='./data', train=False, download=True, transform=transform_plain)
        loader_norm = DataLoader(dataset_norm, batch_size=1, shuffle=False)
        loader_plain = DataLoader(dataset_plain, batch_size=1, shuffle=False)

        found_sample = False
        for (img_norm, lbl), (img_plain, _) in zip(loader_norm, loader_plain):
            # 只检查我们感兴趣的类别
            if lbl.item() == class_to_find:
                img_norm, lbl = img_norm.to(DEVICE), lbl.to(DEVICE)
                output = model(img_norm)
                pred = torch.argmax(output, dim=1)
                
                # 确保模型最初能正确分类
                if pred.item() == lbl.item():
                    loss = criterion(output, lbl)
                    if loss.item() > max_loss:
                        found_sample = True
                        max_loss = loss.item()
                        hardest_sample_normalized = img_norm.cpu()
                        hardest_label = lbl.cpu()
                        original_image_unnormalized = img_plain.cpu() 
    
    if found_sample:
        print(f"Search complete! Found hardest '{class_to_find}' with initial loss: {max_loss:.4f}")
        return original_image_unnormalized, hardest_sample_normalized, hardest_label
    else:
        raise ValueError(f"Could not find any correctly classified sample for class '{class_to_find}'. The model might be poorly trained.")


# --- 5. (已修改) 主执行流程 ---
if __name__ == "__main__":
    # 训练模型
    model = train_model()

    # --- 定义要搜索的类别和攻击目标 ---
    class_to_find = 1
    target_class = 2
    
    # 寻找类别为1的最难样本
    criterion_search = nn.CrossEntropyLoss()
    x_original_unnormalized, x_normalized, y_original = find_hardest_sample_for_class(model, criterion_search, class_to_find=class_to_find)

    # 定义攻击目标为2
    y_target = torch.tensor([target_class])
    print(f"\nOriginal label: {y_original.item()}, Target label: {y_target.item()}")
    
    # 执行目标性攻击
    perturbed_normalized = wgf_sampler_targeted(x_normalized, y_original, y_target, model, epoch=9)

    # 反归一化以便于可视化
    unorm = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
    perturbed_image = unorm(perturbed_normalized)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # --- 检查模型预测并可视化 ---
    model.to(DEVICE).eval()
    with torch.no_grad():
        original_output = model(x_normalized.to(DEVICE))
        perturbed_output = model(perturbed_normalized.to(DEVICE))

    original_pred = torch.argmax(original_output, dim=1)
    perturbed_pred = torch.argmax(perturbed_output, dim=1)

    print(f"\nModel Prediction on Original: {original_pred.item()}")
    print(f"Model Prediction on Perturbed: {perturbed_pred.item()}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.25))

    axes[0].imshow(x_original_unnormalized.squeeze(), cmap='gray')
    axes[0].set_title(f'Original Image, Prediction: {original_pred.item()}')
    axes[0].axis('off')

    axes[1].imshow(perturbed_image.squeeze().cpu().numpy(), cmap='gray')
    axes[1].set_title(f'Perturbed Image, Prediction: {perturbed_pred.item()}')
    axes[1].axis('off')

    # perturbation = (perturbed_image.cpu() - x_original_unnormalized).abs()
    # axes[2].imshow(perturbation.squeeze().numpy(), cmap='gray')
    # axes[2].set_title('The Perturbation (Noise)')
    # axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('mnist_targeted_wgf_attack.png')
    plt.show()