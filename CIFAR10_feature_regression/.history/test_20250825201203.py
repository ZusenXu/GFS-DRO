import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Union

# Note: scikit-learn is a common dependency for these metrics.
# If not installed, run: pip install scikit-learn
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    print("scikit-learn not found. Accuracy will not be calculated in the DRO classes.")
    print("Please run: pip install scikit-learn")
    accuracy_score = None

from scipy.spatial.distance import pdist, squareform

# ==============================================================================
# 1. Global Settings and Helper Classes
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DROError(Exception):
    pass

class LinearModel(nn.Module):
    """A simple linear model for logistic regression."""
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class PaperNegativeLogLikelihoodLoss(nn.Module):
    """
    Implements the negative log-likelihood loss from the paper:
    h_B(x, y) = -y^T B^T x + log(1^T e^(B^T x))
    This is mathematically equivalent to PyTorch's nn.CrossEntropyLoss.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        log_sum_exp = torch.logsumexp(logits, dim=1)
        true_class_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        loss_per_sample = log_sum_exp - true_class_logits
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else: # 'none'
            return loss_per_sample

# ==============================================================================
# 2. Base Class for DRO Models
# ==============================================================================

class BaseLinearDRO:
    """Base class for all DRO linear models, containing common methods."""
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fit_intercept = fit_intercept
        self.device = torch.device("cpu")
        self.model: nn.Module

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1: X = X.reshape(-1, self.input_dim)
        if y.ndim > 1: y = y.flatten()
        if X.shape[0] != y.shape[0]:
            raise DROError(f"Inputs X and y must have the same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")
        if X.shape[1] != self.input_dim:
            raise DROError(f"Expected input feature dimension {self.input_dim}, but got {X.shape[1]}")
        return X, y

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        X_tensor = self._to_tensor(X)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# ==============================================================================
# 3. RGO and SVGD Model Implementations
# ==============================================================================

class SinkhornDROLogisticRGO(BaseLinearDRO):
    """Sinkhorn DRO using Robust Gradient Optimization (RGO) to find worst-case samples."""
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1,
                 lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 5,
                 num_samples: int = 1, max_iter: int = 30, learning_rate: float = 0.001,
                 batch_size: int = 128, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate, self.batch_size = epsilon, lambda_param, num_samples, max_iter, learning_rate, batch_size
        self.rgo_inner_lr, self.rgo_inner_steps = rgo_inner_lr, rgo_inner_steps
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)

    def _get_model_loss_scalar_for_grad(self, x_features: torch.Tensor, y_target: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        predictions_logits = model_instance(x_features)
        return PaperNegativeLogLikelihoodLoss(reduction='sum')(predictions_logits, y_target)

    def _rgo_sampler(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor,
                     current_model_state: nn.Module) -> torch.Tensor:
        x_orig_detached = x_original_batch.detach()
        x_pert = x_orig_detached.clone()
        
        for _ in range(self.rgo_inner_steps):
            x_pert.requires_grad_(True)
            f_model_loss = self._get_model_loss_scalar_for_grad(x_pert, y_original_batch, current_model_state)
            grad_f_model, = torch.autograd.grad(f_model_loss, x_pert, retain_graph=False)
            x_pert = x_pert.detach()
            grad_total = grad_f_model / self.lambda_param - 2 * (x_pert - x_orig_detached)
            x_pert += self.rgo_inner_lr * grad_total
        
        return x_pert

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        for epoch in range(self.max_iter):
            pbar_desc = f"Training RGO (inner_steps={self.rgo_inner_steps}) Epoch {epoch+1}/{self.max_iter}"
            pbar = tqdm(dataloader, desc=pbar_desc)
            for x_original_batch, y_original_batch in pbar:
                self.model.eval()
                x_rgo_batch = self._rgo_sampler(x_original_batch, y_original_batch, self.model)
                
                self.model.train()
                predictions_logits_batch = self.model(x_rgo_batch)
                
                criterion = PaperNegativeLogLikelihoodLoss(reduction='mean')
                loss = criterion(predictions_logits_batch, y_original_batch)

                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_theta.step()
                pbar.set_postfix(loss=loss.item())

class SVGD:
    """Implementation of Stein Variational Gradient Descent."""
    def _svgd_kernel(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N, D = theta.shape
        pairwise_sq_dists = squareform(pdist(theta, 'sqeuclidean'))
        h2 = 0.5 * np.median(pairwise_sq_dists) / (np.log(N + 1) + 1e-8)
        h2 = np.max([h2, 1e-6])
        K = np.exp(-pairwise_sq_dists / (2 * h2))
        grad_K = -(K @ theta - np.sum(K, axis=1, keepdims=True) * theta) / h2
        return K, grad_K

    def update(self, x0: np.ndarray, grad_log_prob: callable, n_iter: int = 10, stepsize: float = 1e-2, alpha: float = 0.9) -> np.ndarray:
        theta = np.copy(x0)
        fudge_factor = 1e-6
        historical_grad = np.zeros_like(theta)
        for i in range(n_iter):
            lnpgrad = grad_log_prob(theta)
            k, grad_k = self._svgd_kernel(theta)
            grad_theta = (k @ lnpgrad + grad_k) / x0.shape[0]
            if i == 0:
                historical_grad = grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = grad_theta / (fudge_factor + np.sqrt(historical_grad))
            theta += stepsize * adj_grad
        return theta

class SinkhornDROLogisticSVGD(BaseLinearDRO):
    """Sinkhorn DRO using SVGD to find worst-case samples."""
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1,
                 lambda_param: float = 1.0, svgd_n_iter: int = 5, svgd_stepsize: float = 1e-2,
                 num_samples: int = 10, max_iter: int = 30, learning_rate: float = 0.001,
                 batch_size: int = 128, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate, self.batch_size = epsilon, lambda_param, num_samples, max_iter, learning_rate, batch_size
        self.svgd_n_iter = svgd_n_iter
        self.svgd_stepsize = svgd_stepsize
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
        self.svgd_computer = SVGD()

    def _svgd_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module) -> torch.Tensor:
        def grad_log_prob(x_np: np.ndarray) -> np.ndarray:
            grad_prior = -2.0 / self.epsilon * (x_np - x_orig.cpu().numpy())
            x_torch = self._to_tensor(x_np).requires_grad_(True)
            y_torch = y_orig.repeat(x_torch.shape[0])
            predictions = model(x_torch)
            loss = PaperNegativeLogLikelihoodLoss(reduction='sum')(predictions, y_torch)
            grad_likelihood = torch.autograd.grad(loss, x_torch, retain_graph=True)[0]
            return grad_prior - (grad_likelihood / (self.lambda_param * self.epsilon)).detach().cpu().numpy()

        mean = x_orig
        std_dev = torch.sqrt(torch.tensor(self.epsilon / 2.0, device=self.device))
        X0_torch = mean + std_dev * torch.randn(self.num_samples, self.input_dim, device=self.device)
        final_particles_np = self.svgd_computer.update(X0_torch.cpu().numpy(), grad_log_prob, self.svgd_n_iter, self.svgd_stepsize)
        return self._to_tensor(final_particles_np)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        for epoch in range(self.max_iter):
            pbar_desc = f"Training SVGD (n_iter={self.svgd_n_iter}) Epoch {epoch+1}/{self.max_iter}"
            pbar = tqdm(dataloader, desc=pbar_desc)
            for x_batch, y_batch in pbar:
                batch_worst_case_samples = []
                self.model.eval()
                for i in range(x_batch.size(0)):
                    x_orig, y_orig = x_batch[i:i+1], y_batch[i]
                    worst_case_samples = self._svgd_sampler(x_orig, y_orig, self.model)
                    batch_worst_case_samples.append(worst_case_samples)
                
                all_worst_samples = torch.cat(batch_worst_case_samples, dim=0)
                y_repeated = y_batch.repeat_interleave(self.num_samples, dim=0)

                self.model.train()
                predictions = self.model(all_worst_samples)
                
                criterion = PaperNegativeLogLikelihoodLoss(reduction='mean')
                loss = criterion(predictions, y_repeated)
                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_theta.step()
                pbar.set_postfix(loss=loss.item())

# ==============================================================================
# 4. Data Loading and Evaluation Functions
# ==============================================================================

def get_cifar10_dataloaders(batch_size=128):
    """Downloads and prepares the DataLoader for the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # FIX: Corrected the standard deviation for the third channel from 2010 to 0.2010.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def extract_features(data_loader, device):
    """Extracts features using a pre-trained ResNet-18 model."""
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            features = feature_extractor(images).view(images.size(0), -1)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)

def pgd_attack(model, features, labels, epsilon, alpha, num_iter):
    """PGD Adversarial Attack (l2 norm)."""
    perturbed_features = features.clone().detach().to(DEVICE)
    perturbed_features.requires_grad = True
    original_features = features.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)
    criterion = PaperNegativeLogLikelihoodLoss()
    for _ in range(num_iter):
        model.zero_grad()
        outputs = model(perturbed_features)
        loss = criterion(outputs, labels)
        loss.backward()
        grad = perturbed_features.grad.detach()
        
        # l2 attack: move along the gradient direction
        grad_norm = torch.linalg.norm(grad.view(grad.shape[0], -1), dim=1) + 1e-12
        grad_unit = grad / grad_norm.view(-1, 1)
        perturbed_features.data = perturbed_features.data + alpha * grad_unit
        
        # Project perturbation back to the l2 ball
        perturbation = perturbed_features.data - original_features.data
        norm = torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1)
        factor = epsilon / (norm + 1e-12)
        factor = torch.min(torch.ones_like(norm), factor)
        
        perturbation = perturbation * factor.view(-1, 1)
        perturbed_features.data = original_features.data + perturbation

    return perturbed_features.detach()

def evaluate_robustness(model, test_features_tensor, test_labels_tensor, epsilon_abs):
    """Evaluates model robustness at a specific attack strength."""
    model.to(DEVICE)
    model.eval()
    
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    correct, total = 0, 0
    for features, labels in test_loader:
        perturbed_features = pgd_attack(model=model, features=features, labels=labels, 
                                        epsilon=epsilon_abs, alpha=epsilon_abs / 4, num_iter=10)
        with torch.no_grad():
            outputs = model(perturbed_features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()
            
    return 100 * correct / total

# ==============================================================================
# 5. Main Execution Flow: Experiment, Data Collection, and Plotting
# ==============================================================================

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print("Loading data and extracting features...")
    # To speed up demonstrations, we can cache the extracted features
    try:
        train_features = torch.load('train_features.pt')
        train_labels = torch.load('train_labels.pt')
        test_features = torch.load('test_features.pt')
        test_labels = torch.load('test_labels.pt')
        print("Loaded features from cache.")
    except FileNotFoundError:
        print("Cached features not found. Extracting from scratch...")
        cifar_train_loader, cifar_test_loader = get_cifar10_dataloaders()
        train_features, train_labels = extract_features(cifar_train_loader, DEVICE)
        test_features, test_labels = extract_features(cifar_test_loader, DEVICE)
        torch.save(train_features, 'train_features.pt')
        torch.save(train_labels, 'train_labels.pt')
        torch.save(test_features, 'test_features.pt')
        torch.save(test_labels, 'test_labels.pt')

    train_features_np, train_labels_np = train_features.numpy(), train_labels.numpy()
    
    INPUT_DIM = train_features_np.shape[1]
    NUM_CLASSES = 10
    
    # --- Experiment Settings ---
    # Define the range of hyperparameters to test
    rgo_steps_range = [1, 3, 5, 10, 15]
    svgd_iter_range = [1, 3, 5, 10, 15]
    
    # Define a fixed PGD attack strength for evaluation
    avg_feature_norm = np.mean(np.linalg.norm(test_features.numpy(), axis=1))
    EVAL_PERTURBATION_LEVEL = 0.015 
    EVAL_EPSILON_ABS = EVAL_PERTURBATION_LEVEL * avg_feature_norm
    
    results_data = []

    # --- RGO Experiment ---
    for steps in rgo_steps_range:
        print(f"\n--- Training RGO Model with rgo_inner_steps = {steps} ---")
        rgo_model = SinkhornDROLogisticRGO(
            INPUT_DIM, NUM_CLASSES, device=DEVICE, 
            rgo_inner_steps=steps,
            max_iter=30 # Reducing epochs for a quicker demonstration
        )
        
        start_time = time.time()
        rgo_model.fit(train_features_np, train_labels_np)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        print("Evaluating RGO model...")
        robust_acc = evaluate_robustness(rgo_model.model, test_features, test_labels, EVAL_EPSILON_ABS)
        
        results_data.append({
            'model': 'RGO',
            'hyperparameter': steps,
            'training_time': training_time,
            'robust_accuracy': robust_acc
        })
        print(f"Result: steps={steps}, time={training_time:.2f}s, robust_acc={robust_acc:.2f}%")

    # --- SVGD Experiment ---
    for n_iter in svgd_iter_range:
        print(f"\n--- Training SVGD Model with svgd_n_iter = {n_iter} ---")
        svgd_model = SinkhornDROLogisticSVGD(
            INPUT_DIM, NUM_CLASSES, device=DEVICE,
            svgd_n_iter=n_iter,
            max_iter=5 # Reducing epochs for a quicker demonstration
        )
        
        start_time = time.time()
        svgd_model.fit(train_features_np, train_labels_np)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        print("Evaluating SVGD model...")
        robust_acc = evaluate_robustness(svgd_model.model, test_features, test_labels, EVAL_EPSILON_ABS)
        
        results_data.append({
            'model': 'SVGD',
            'hyperparameter': n_iter,
            'training_time': training_time,
            'robust_accuracy': robust_acc
        })
        print(f"Result: n_iter={n_iter}, time={training_time:.2f}s, robust_acc={robust_acc:.2f}%")
        
    # --- Save Results ---
    df_results = pd.DataFrame(results_data)
    df_results.to_csv("rgo_vs_svgd_comparison.csv", index=False)
    print("\nExperiment results saved to rgo_vs_svgd_comparison.csv")
    print(df_results)

    # --- Plotting ---
    print("\nGenerating plots...")
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Comparison of RGO and SVGD Efficiency', fontsize=20)

    # Plot 1: Robust Accuracy vs. Training Time
    sns.lineplot(data=df_results, x='training_time', y='robust_accuracy', hue='model', marker='o', ax=axes[0])
    axes[0].set_title('Robust Accuracy vs. Training Time')
    axes[0].set_xlabel('Training Time (seconds)')
    axes[0].set_ylabel(f'Robust Accuracy (%) at PGD eps={EVAL_PERTURBATION_LEVEL:.3f}')

    # Plot 2: Robust Accuracy vs. Hyperparameter
    sns.lineplot(data=df_results, x='hyperparameter', y='robust_accuracy', hue='model', marker='o', ax=axes[1])
    axes[1].set_title('Robust Accuracy vs. Inner Iterations')
    axes[1].set_xlabel('Number of Inner Iterations (steps/n_iter)')
    axes[1].set_ylabel('Robust Accuracy (%)')
    
    # Plot 3: Training Time vs. Hyperparameter
    sns.lineplot(data=df_results, x='hyperparameter', y='training_time', hue='model', marker='o', ax=axes[2])
    axes[2].set_title('Training Time vs. Inner Iterations')
    axes[2].set_xlabel('Number of Inner Iterations (steps/n_iter)')
    axes[2].set_ylabel('Training Time (seconds)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("rgo_vs_svgd_comparison.png")
    print("Plots saved to rgo_vs_svgd_comparison.png")
    # plt.show()
