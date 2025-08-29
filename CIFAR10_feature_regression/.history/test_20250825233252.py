import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import math
import os
from typing import Tuple, Dict, List, Union

# --- Imports from your provided code ---
from scipy.spatial.distance import pdist, squareform
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    print("scikit-learn not found. Accuracy will not be calculated in the DRO classes.")
    print("Please run: pip install scikit-learn")
    accuracy_score = None

# --- Imports for plotting ---
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. Global Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. Data Loading and Feature Extraction (Replaced) ---

def get_cifar10_dataloaders(batch_size=128):
    """(Replaced) Downloads and prepares the DataLoader for the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def extract_features(data_loader, model, device):
    """(Replaced) Extracts features using a pre-trained ResNet-50 model."""
    model.to(device)
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


# --- 3. Adversarial Attack (Replaced) ---

def pgd_attack(model, features, labels, epsilon, alpha, num_iter):
    """(Replaced) PGD Adversarial Attack (l-infinity norm)"""
    features, labels = features.to(DEVICE), labels.to(DEVICE)
    perturbed_features = features.clone().detach().requires_grad_(True)
    original_features = features.clone().detach()

    for _ in range(num_iter):
        model.zero_grad()
        outputs = model(perturbed_features)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        grad = perturbed_features.grad.detach()
        perturbed_features.data = perturbed_features.data + alpha * grad.sign()
        
        eta = torch.clamp(perturbed_features.data - original_features, min=-epsilon, max=epsilon)
        perturbed_features.data = torch.clamp(original_features + eta, min=0, max=1).detach()
        perturbed_features.requires_grad_(True)

    return perturbed_features.detach()

# --- 4. Model ---

# (Replaced) Logistic Regression for SAA
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, weight, bias=False):
        super(LogisticRegression, self).__init__()
        self.device = DEVICE
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        if weight is not None:
             self.linear.weight = nn.Parameter(weight.to(self.device))
    
    def forward(self, x):
        return self.linear(x.to(self.device))
    
    def cross_entropy_metric(self, predictions, targets):
        return nn.functional.cross_entropy(predictions, targets.to(self.device))

# --- DRO Model Definitions ---

class DROError(Exception):
    pass

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class BaseLinearDRO:
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
            raise DROError(f"X and y must have the same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")
        if X.shape[1] != self.input_dim:
            raise DROError(f"Expected input_dim={self.input_dim} features for X, got {X.shape[1]}")
        return X, y

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        X_tensor = self._to_tensor(X)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_val, _ = self._validate_inputs(X, np.zeros(X.shape[0]))
        X_tensor = self._to_tensor(X_val)
        self.model.eval()
        with torch.no_grad():
            predictions_logits = self.model(X_tensor).cpu().numpy()
        return np.argmax(predictions_logits, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if accuracy_score is None: return -1.0
        y_pred = self.predict(X)
        return accuracy_score(y.flatten(), y_pred)
    
    def _compute_loss(self, predictions, targets, m, lambda_reg):
        criterion = nn.CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg

# --- (UNCHANGED) SinkhornLinearDRO ---
class SinkhornLinearDRO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 1e-3,
                 lambda_param: float = 1e2, max_iter: int = 100, learning_rate: float = 1e-2,
                 num_samples: int = 32, batch_size: int = 64, device: str = "cpu",
                 checkpoint_dir: str = "./checkpoints", interval: int = 10):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.max_iter, self.learning_rate, self.num_samples, self.batch_size = epsilon, lambda_param, max_iter, learning_rate, num_samples, batch_size
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(input_dim, output_dim=num_classes, bias=fit_intercept).to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        
        self.model.train()
        for epoch in range(1, self.max_iter + 1):
            pbar = tqdm(dataloader, desc=f"Training SinkhornLinearDRO Epoch {epoch}/{self.max_iter}")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                m = self.num_samples
                
                expanded_data = data.repeat_interleave(m, dim=0)
                noise = torch.randn_like(expanded_data) * math.sqrt(self.epsilon)
                noisy_data = expanded_data + noise
                repeated_target = target.repeat_interleave(m, dim=0)
                
                predictions = self.model(noisy_data)
                loss = self._compute_loss(predictions, repeated_target, m, lambda_reg)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

            if epoch % self.interval == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{self.checkpoint_dir}/Dual_model_epoch_{epoch}.ckpt"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved Dual model checkpoint to {checkpoint_path}")


class SVGD:
    def _svgd_kernel(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N, D = theta.shape
        pairwise_sq_dists = squareform(pdist(theta, 'sqeuclidean'))
        h2 = 0.5 * np.median(pairwise_sq_dists) / (np.log(N + 1) + 1e-8)
        h2 = np.max([h2, 1e-6])
        K = np.exp(-pairwise_sq_dists / (2 * h2))
        grad_K = -(K @ theta - np.sum(K, axis=1, keepdims=True) * theta) / h2
        return K, grad_K

    def update(self, x0: np.ndarray, grad_log_prob: callable, n_iter: int = 50, stepsize: float = 1e-2, alpha: float = 0.9) -> np.ndarray:
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
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1,
                 lambda_param: float = 1.0, svgd_n_iter: int = 50, svgd_stepsize: float = 1e-2,
                 num_samples: int = 10, max_iter: int = 30, learning_rate: float = 0.01,
                 batch_size: int = 64, device: str = "cpu",
                 checkpoint_dir: str = "./checkpoints", interval: int = 10):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate, self.batch_size = epsilon, lambda_param, num_samples, max_iter, learning_rate, batch_size
        self.svgd_n_iter, self.svgd_stepsize = svgd_n_iter, svgd_stepsize
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
        self.svgd_computer = SVGD()
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval

    def _svgd_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch: int) -> torch.Tensor:
        def grad_log_prob(x_np: np.ndarray) -> np.ndarray:
            grad_prior = -2.0 / self.epsilon * (x_np - x_orig.cpu().numpy())
            x_torch = self._to_tensor(x_np).requires_grad_(True)
            y_torch = y_orig.repeat(x_torch.shape[0])
            predictions = model(x_torch)
            loss = nn.CrossEntropyLoss(reduction='sum')(predictions, y_torch)
            grad_likelihood = torch.autograd.grad(loss, x_torch, retain_graph=True)[0]
            return grad_prior - (grad_likelihood / (self.lambda_param * self.epsilon)).detach().cpu().numpy()

        mean = x_orig
        std_dev = torch.sqrt(torch.tensor(self.epsilon / 2.0, device=self.device))
        X0_torch = mean + std_dev * torch.randn(self.num_samples, self.input_dim, device=self.device)
        final_particles_np = self.svgd_computer.update(X0_torch.cpu().numpy(), grad_log_prob, int(min(5, self.svgd_n_iter * (epoch + 1) / self.max_iter)), self.svgd_stepsize)
        return self._to_tensor(final_particles_np)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        for epoch in range(1, self.max_iter + 1):
            pbar = tqdm(dataloader, desc=f"Training SVGD Epoch {epoch}/{self.max_iter}")
            for x_batch, y_batch in pbar:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                batch_worst_case_samples = []
                self.model.eval()
                for i in range(x_batch.size(0)):
                    x_orig, y_orig = x_batch[i:i+1], y_batch[i]
                    worst_case_samples = self._svgd_sampler(x_orig, y_orig, self.model, epoch)
                    batch_worst_case_samples.append(worst_case_samples)
                
                all_worst_samples = torch.cat(batch_worst_case_samples, dim=0)
                y_repeated = y_batch.repeat_interleave(self.num_samples, dim=0)

                self.model.train()
                predictions = self.model(all_worst_samples)
                loss = nn.CrossEntropyLoss(reduction='mean')(predictions, y_repeated)
                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_theta.step()
                pbar.set_postfix(loss=loss.item())

            if epoch % self.interval == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{self.checkpoint_dir}/Transport_model_epoch_{epoch}.ckpt"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved Transport model checkpoint to {checkpoint_path}")


class SinkhornDROLogisticRGO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1,
                 lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 20,
                 num_samples: int = 10, max_iter: int = 30, learning_rate: float = 0.01,
                 batch_size: int = 64, device: str = "cpu",
                 checkpoint_dir: str = "./checkpoints", interval: int = 10):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate, self.batch_size = epsilon, lambda_param, num_samples, max_iter, learning_rate, batch_size
        self.rgo_inner_lr, self.rgo_inner_steps = rgo_inner_lr, rgo_inner_steps
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval


    def _get_model_loss_value_scalar_for_grad(self, x_features: torch.Tensor, y_target: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        predictions = model_instance(x_features)
        return nn.CrossEntropyLoss(reduction='sum')(predictions, y_target)

    def _rgo_sampler_vectorized(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor,
                                current_model_state: nn.Module) -> torch.Tensor:
        x_orig_detached = x_original_batch.detach()
        x_pert = x_orig_detached.clone()
        
        for _ in range(self.rgo_inner_steps):
            x_pert.requires_grad_(True)
            f_model_loss = self._get_model_loss_value_scalar_for_grad(x_pert, y_original_batch, current_model_state)
            grad_f_model, = torch.autograd.grad(f_model_loss, x_pert, retain_graph=False)
            x_pert = x_pert.detach()
            grad_total = grad_f_model / self.lambda_param - 2 * (x_pert - x_orig_detached)
            x_pert += self.rgo_inner_lr * grad_total
        
        x_opt_star_batch = x_pert
        var_rgo = self.epsilon
        if var_rgo <= 1e-12:
            return x_opt_star_batch.repeat_interleave(self.num_samples, dim=0)

        std_rgo = math.sqrt(var_rgo)
        noise = torch.randn(x_opt_star_batch.size(0), self.num_samples, self.input_dim, device=self.device) * std_rgo
        final_samples = x_opt_star_batch.unsqueeze(1) + noise
        return final_samples.view(-1, self.input_dim)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        for epoch in range(1, self.max_iter + 1):
            pbar = tqdm(dataloader, desc=f"Training RGO Epoch {epoch}/{self.max_iter}")
            for x_original_batch, y_original_batch in pbar:
                x_original_batch, y_original_batch = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_rgo_batch = self._rgo_sampler_vectorized(x_original_batch, y_original_batch, self.model)
                
                self.model.train()
                y_repeated_batch = y_original_batch.repeat_interleave(self.num_samples, dim=0)
                predictions_logits_batch = self.model(x_rgo_batch)
                
                loss = nn.CrossEntropyLoss(reduction='mean')(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_theta.step()
                pbar.set_postfix(loss=loss.item())
            
            if epoch % self.interval == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{self.checkpoint_dir}/RGO_model_epoch_{epoch}.ckpt"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved RGO model checkpoint to {checkpoint_path}")


# --- 5. Training and Evaluation ---

# (Replaced)
def train_saa(model, lr, train_dataloader, num_epochs, checkpoint_dir, interval=5):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Training SAA Epoch {epoch}/{num_epochs}")
        for features, labels in pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = model.cross_entropy_metric(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        if epoch % interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/SAA_model_epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved SAA model checkpoint to {checkpoint_path}")

# (Replaced)
def validate_model_at_intervals(model_obj, test_features_np, test_labels_np, checkpoint_dir, train_method, max_iter, interval):
    print(f"\n--- Validating Model {train_method} ---")
    model = model_obj.model
    model.to(DEVICE)
    results = {}
    for epoch in range(interval, max_iter + 1, interval):
        checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file not found, skipping epoch {epoch}: {checkpoint_path}")
            continue
        
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        accuracy = model_obj.score(test_features_np, test_labels_np) * 100
        print(f"Epoch {epoch}: Accuracy on clean test set: {accuracy:.2f}%")
        results[epoch] = accuracy
    return results

def evaluate_robustness_final_model(model_obj, test_features, test_labels, attack_fn, epsilon_list, checkpoint_path):
    model = model_obj.model
    model.to(DEVICE)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Final model checkpoint not found: {checkpoint_path}")
        return {eps: 0 for eps in [0.0] + epsilon_list}

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    results = {}
    clean_accuracy = model_obj.score(test_features.numpy(), test_labels.numpy()) * 100
    print(f"Final model accuracy on clean test set: {clean_accuracy:.2f}%")
    results[0.0] = clean_accuracy
    
    test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=128)

    for epsilon in epsilon_list:
        correct, total = 0, 0
        pbar_desc = f"Evaluating robustness (epsilon={epsilon:.4f})"
        for features, labels in tqdm(test_loader, desc=pbar_desc):
            perturbed_features = attack_fn(model=model, features=features, labels=labels, epsilon=epsilon, alpha=epsilon / 8, num_iter=10)
            with torch.no_grad():
                outputs = model(perturbed_features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(DEVICE)).sum().item()
        perturbed_accuracy = 100 * correct / total
        print(f"Accuracy under PGD attack (epsilon={epsilon:.4f}): {perturbed_accuracy:.2f}%")
        results[epsilon] = perturbed_accuracy
    return results


# --- Plotting Function ---
def plot_results(results_dict, perturbation_levels):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    for model_name, results in results_dict.items():
        # Ensure results are sorted by epsilon to plot correctly
        sorted_epsilons = sorted(results.keys())
        # Map perturbation levels to the sorted epsilon results
        y_values = [100.0 - results.get(eps, 0) for eps in sorted_epsilons]
        plt.plot(perturbation_levels, y_values, marker='o', linestyle='--', label=model_name)
    plt.xlabel("Perturbation Level (Epsilon)")
    plt.ylabel("Misclassification Rate (%)")
    plt.title("Model Robustness under PGD Attack on CIFAR-10 Features")
    plt.legend(title="Model")
    plt.grid(True, which='both', linestyle='-')
    plt.ylim(bottom=0)
    plt.xticks(perturbation_levels)
    plt.savefig("robustness_comparison_plot.png")
    print("\nPlot saved as 'robustness_comparison_plot.png'")


# --- 6. Main Execution Flow ---
if __name__ == '__main__':
    CHECKPOINT_DIR = "./checkpoints"
    MAX_ITER = 30
    INTERVAL = 10
    FEATURE_FILE = "cifar10_resnet50_features.pth"

    # --- Feature Extraction or Loading ---
    if os.path.exists(FEATURE_FILE):
        print(f"Loading features from {FEATURE_FILE}...")
        feature_data = torch.load(FEATURE_FILE)
        train_features = feature_data["train_features"]
        train_labels = feature_data["train_labels"]
        test_features = feature_data["test_features"]
        test_labels = feature_data["test_labels"]
    else:
        print("Feature file not found. Extracting features...")
        resnet_feature_extractor = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        resnet_feature_extractor.fc = nn.Sequential(
            nn.Linear(resnet_feature_extractor.fc.in_features, 250),
            nn.ReLU(inplace=True)
        )
        cifar_train_loader, cifar_test_loader = get_cifar10_dataloaders()
        train_features, train_labels = extract_features(cifar_train_loader, resnet_feature_extractor, DEVICE)
        test_features, test_labels = extract_features(cifar_test_loader, resnet_feature_extractor, DEVICE)
        
        print(f"Saving features to {FEATURE_FILE}...")
        torch.save({
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
        }, FEATURE_FILE)

    train_features_np, train_labels_np = train_features.numpy(), train_labels.numpy()
    test_features_np, test_labels_np = test_features.numpy(), test_labels.numpy()
    
    INPUT_DIM = train_features_np.shape[1]
    NUM_CLASSES = 10
    
    x0 = torch.nn.init.normal_(torch.empty((NUM_CLASSES, INPUT_DIM)), mean=0.0, std=math.sqrt(0.2))

    # --- SAA ---
    print("\n--- Training SAA Model ---")
    saa_model = LogisticRegression(INPUT_DIM, NUM_CLASSES, x0.clone().detach()).to(DEVICE)
    feature_train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=128, shuffle=True)
    train_saa(saa_model, lr=3e-2, train_dataloader=feature_train_loader, num_epochs=MAX_ITER, checkpoint_dir=CHECKPOINT_DIR, interval=INTERVAL)
    
    # --- DRO Models ---
    dro_models = {
        "Dual": SinkhornLinearDRO(INPUT_DIM, NUM_CLASSES, device=DEVICE, max_iter=MAX_ITER, interval=INTERVAL, checkpoint_dir=CHECKPOINT_DIR),
        "Transport": SinkhornDROLogisticSVGD(INPUT_DIM, NUM_CLASSES, device=DEVICE, max_iter=MAX_ITER, interval=INTERVAL, checkpoint_dir=CHECKPOINT_DIR),
        "RGO": SinkhornDROLogisticRGO(INPUT_DIM, NUM_CLASSES, device=DEVICE, max_iter=MAX_ITER, interval=INTERVAL, checkpoint_dir=CHECKPOINT_DIR)
    }
    for name, model_obj in dro_models.items():
        print(f"\n--- Training {name} Model ---")
        model_obj.fit(train_features_np, train_labels_np)

    # --- Validate All Models ---
    class SAAWrapper(BaseLinearDRO):
        def __init__(self, model, input_dim, num_classes):
            super().__init__(input_dim, num_classes, True)
            self.model = model
    
    all_models_for_validation = {
        "SAA": SAAWrapper(saa_model, INPUT_DIM, NUM_CLASSES),
        **dro_models
    }

    for name, model_obj in all_models_for_validation.items():
        validate_model_at_intervals(model_obj, test_features_np, test_labels_np, CHECKPOINT_DIR, name, MAX_ITER, INTERVAL)

    # --- Evaluate Final Model Robustness ---
    print("\n--- Evaluating Final Model Robustness ---")
    avg_feature_norm = np.mean(np.linalg.norm(test_features_np, axis=1))
    print(f"Average L2 norm of test set features: {avg_feature_norm:.2f}")
    perturbation_levels = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    epsilon_values = [level * avg_feature_norm for level in perturbation_levels]
    
    all_robustness_results = {}
    for name, model_obj in all_models_for_validation.items():
        print(f"\n--- Evaluating Robustness of {name} ---")
        final_checkpoint_path = f"{CHECKPOINT_DIR}/{name}_model_epoch_{MAX_ITER}.ckpt"
        
        results = evaluate_robustness_final_model(model_obj, test_features, test_labels, pgd_attack, epsilon_values[1:], final_checkpoint_path)
        all_robustness_results[name] = results

    # --- Print Summary and Plot ---
    print("\n\n--- FINAL ROBUSTNESS RESULTS SUMMARY ---")
    header = f"{'Model':<12} | " + " | ".join([f"Acc. @ eps={p_level:.3f}" for p_level in perturbation_levels])
    print(header)
    print("-" * len(header))
    for name, results in all_robustness_results.items():
        row = f"{name:<12} | "
        acc_values = [f"{results.get(eps_val, 0):>6.2f}%" for eps_val in [0.0] + epsilon_values[1:]]
        row += " | ".join(acc_values)
        print(row)
    
    print("\nGenerating results plot...")
    plot_results_data = {}
    for name, results in all_robustness_results.items():
        plot_results_data[name] = {level: results.get(level * avg_feature_norm, results.get(0.0)) for level in perturbation_levels}
        plot_results_data[name][0.0] = results[0.0]

    plot_results(plot_results_data, perturbation_levels)