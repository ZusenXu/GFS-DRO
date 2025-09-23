import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define your function and global variables (Unchanged) ---
NUM_SAMPLES_PER_POINT = 1
LAMBDA_PARAM = 0.1
EPSILON = 0.01
MAX_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inner_lr = 1e-2
inner_steps = 300

def wgf_sampler(x_original_batch, y_original_batch, model, epoch, lr=inner_lr, inner_steps=inner_steps):
    """
    WGF Sampler: Uses Langevin dynamics to sample from the worst-case distribution.
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

# --- 2. Prepare a pre-trained MNIST classification model (Unchanged) ---
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

model = train_model()

# --- 3. (Corrected) Search for the "hardest" sample ---

# CHANGE 1: Removed the unused 'loader' parameter from the function definition
def find_hardest_sample_correctly_classified(model, criterion):
    """
    Iterates through the dataset to find the sample that the model correctly classifies
    but with the highest loss value.
    """
    print("\nSearching for the highest-loss (but still correct) sample in the test set...")
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

        for (img_norm, lbl), (img_plain, _) in zip(loader_norm, loader_plain):
            img_norm, lbl = img_norm.to(DEVICE), lbl.to(DEVICE)
            
            output = model(img_norm)
            pred = torch.argmax(output, dim=1)
            
            if pred.item() == lbl.item():
                loss = criterion(output, lbl)
                if loss.item() > max_loss:
                    max_loss = loss.item()
                    hardest_sample_normalized = img_norm.cpu()
                    hardest_label = lbl.cpu()
                    original_image_unnormalized = img_plain.cpu() 
                    
    print(f"Search complete! Found hardest sample with label {hardest_label.item()}, initial loss: {max_loss:.4f}")
    return original_image_unnormalized, hardest_sample_normalized, hardest_label

criterion = nn.CrossEntropyLoss()
# CHANGE 2: Removed the undefined 'test_loader' from the function call
x_original_unnormalized, x_normalized, y_original = find_hardest_sample_correctly_classified(model, criterion)


# --- 4. Run the perturbation function and visualize (Unchanged) ---
print("\nApplying WGF sampler to the found 'hardest sample'...")
perturbed_normalized = wgf_sampler(x_normalized, y_original, model, epoch=10)

unorm = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))
perturbed_image = unorm(perturbed_normalized)
perturbed_image = torch.clamp(perturbed_image, 0, 1)

# --- 5. Display results (Unchanged) ---
model.to(DEVICE).eval()
with torch.no_grad():
    original_output = model(x_normalized.to(DEVICE))
    perturbed_output = model(perturbed_normalized.to(DEVICE))

original_pred = torch.argmax(original_output, dim=1)
perturbed_pred = torch.argmax(perturbed_output, dim=1)

print(f"\nOriginal Label: {y_original.item()}")
print(f"Model Prediction on Original: {original_pred.item()}")
print(f"Model Prediction on Perturbed: {perturbed_pred.item()}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(x_original_unnormalized.squeeze(), cmap='gray')
axes[0].set_title(f'Hardest Original Image\nLabel: {y_original.item()}\nPrediction: {original_pred.item()}')
axes[0].axis('off')

axes[1].imshow(perturbed_image.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title(f'Perturbed Image\nPrediction: {perturbed_pred.item()}')
axes[1].axis('off')

perturbation = (perturbed_image - x_original_unnormalized).abs()
axes[2].imshow(perturbation.squeeze().cpu().numpy(), cmap='gray')
axes[2].set_title('The Perturbation (Noise)')
axes[2].axis('off')

plt.savefig('mnist_wgf_perturbation.png')
plt.show()