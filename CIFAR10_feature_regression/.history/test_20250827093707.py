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

# --- Imports ---
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    print("scikit-learn not found. Please run: pip install scikit-learn")
    accuracy_score = None
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Global Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. Adversarial Attack (Corrected) ---
def pgd_attack(model, features, labels, epsilon, alpha, num_iter):
    """PGD Adversarial Attack (l2 norm) - CORRECTED VERSION"""
    perturbed_features = features.clone().detach().to(DEVICE)
    perturbed_features.requires_grad = True
    original_features = features.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        if perturbed_features.grad is not None:
            perturbed_features.grad.zero_()

        outputs = model(perturbed_features)
        loss = criterion(outputs, labels)
        loss.backward()

        grad = perturbed_features.grad.detach()
        grad_norm = torch.linalg.norm(grad.view(grad.shape[0], -1), dim=1) + 1e-12
        normalized_grad = grad / grad_norm.view(-1, 1)
        perturbed_features.data = perturbed_features.data + alpha * normalized_grad
        
        perturbation = perturbed_features.data - original_features.data
        norm = torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1)
        factor = epsilon / (norm + 1e-12)
        factor = torch.min(torch.ones_like(norm), factor)
        perturbation = perturbation * factor.view(-1, 1)
        perturbed_features.data = original_features.data + perturbation

    return perturbed_features.detach()


# --- 3. Model Definitions ---
class LogisticRegression(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x): return self.linear(x)

class DROError(Exception): pass

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.linear(x)

class BaseLinearDRO:
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool, sample_level: int = 6):
        self.input_dim, self.num_classes, self.fit_intercept = input_dim, num_classes, fit_intercept
        self.device = torch.device("cpu")
        self.model: nn.Module
        self.sample_level = sample_level
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor: return torch.as_tensor(data, dtype=torch.float32, device=self.device)
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1: X = X.reshape(-1, self.input_dim)
        if y.ndim > 1: y = y.flatten()
        if X.shape[0] != y.shape[0]: raise DROError(f"Shapes mismatch: X {X.shape[0]}, y {y.shape[0]}")
        if X.shape[1] != self.input_dim: raise DROError(f"Input dim mismatch: expected {self.input_dim}, got {X.shape[1]}")
        return X, y
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        dataset = TensorDataset(self._to_tensor(X), torch.as_tensor(y, dtype=torch.long, device=self.device))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_val, _ = self._validate_inputs(X, np.zeros(X.shape[0]))
        self.model.eval()
        with torch.no_grad(): return self.model(self._to_tensor(X_val)).cpu().numpy()
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if accuracy_score is None: return -1.0
        return accuracy_score(y.flatten(), np.argmax(self.predict(X), axis=1))
    def _compute_loss(self, predictions, targets, m, lambda_reg):
        # MODIFIED: Use the custom loss function
        criterion = nn.CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg
    
    def _compute_dual_loss(self, data, targets, lam, epsilon):
        levels = np.arange(self.sample_level + 1)
        numerators = 2.0**(-levels)
        denominator = 2.0 - 2.0**(-self.sample_level)

        probabilities = numerators / denominator
        sampled_level = np.random.choice(levels, p=probabilities)
        m = 2 ** sampled_level
        
        expanded_data = data.repeat_interleave(m, dim=0)
        noise = torch.randn_like(expanded_data) * math.sqrt(epsilon)
        noisy_data = expanded_data + noise
        repeated_target = targets.repeat_interleave(m, dim=0)
        
        predictions = self.model(noisy_data)
        criterion = nn.CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, repeated_target) / max(lam * epsilon, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lam * epsilon
    
class SinkhornBaseDRO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 1e-3,
                 lambda_param: float = 1e2, max_iter: int = 100, learning_rate: float = 1e-2,
                 sample_level: int = 6, batch_size: int = 64, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.max_iter, self.learning_rate, self.sample_level, self.batch_size = epsilon, lambda_param, max_iter, learning_rate, sample_level, batch_size
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(input_dim, output_dim=num_classes, bias=fit_intercept).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        loss_history = []
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Training SinkhornLinearDRO Epoch {epoch+1}/{self.max_iter}")
            for data, target in pbar:
                optimizer.zero_grad()
                levels = np.arange(self.sample_level + 1)
                numerators = 2.0**(-levels)
                denominator = 2.0 - 2.0**(-self.sample_level)
    
                probabilities = numerators / denominator
                sampled_level = np.random.choice(levels, p=probabilities)
                m = 2 ** sampled_level
                
                expanded_data = data.repeat_interleave(m, dim=0)
                noise = torch.randn_like(expanded_data) * math.sqrt(self.epsilon)
                noisy_data = expanded_data + noise
                repeated_target = target.repeat_interleave(m, dim=0)
                
                predictions = self.model(noisy_data)
                loss = self._compute_loss(predictions, repeated_target, m, lambda_reg)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_epoch_loss)
        return loss_history

# class SinkhornBaseDRO(BaseLinearDRO):
#     def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 1.0, lambda_param: float = 1.0, max_iter: int = 100, learning_rate: float = 1e-3, num_samples: int = 10, batch_size: int = 64, noise: float = 0.1, device: str = "cpu"):
#         super().__init__(input_dim, num_classes, fit_intercept)
#         self.epsilon, self.lambda_param, self.max_iter, self.learning_rate = epsilon, lambda_param, max_iter, learning_rate
#         self.num_samples, self.batch_size, self.noise = num_samples, batch_size, noise
#         self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
#         self.model = LinearModel(input_dim, output_dim=num_classes, bias=fit_intercept).to(self.device)
#     def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
#         X, y = self._validate_inputs(X, y)
#         dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
#         optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         self.model.train()
#         loss_history = []
#         for epoch in range(self.max_iter):
#             epoch_loss = 0.0
#             pbar = tqdm(dataloader, desc=f"Training SinkhornBaseDRO Epoch {epoch+1}/{self.max_iter}")
#             for zeta_batch, target_batch in pbar:
#                 optimizer.zero_grad()
#                 batch_size = zeta_batch.size(0)
#                 zeta_batch_expanded = zeta_batch.repeat_interleave(self.num_samples, dim=0).to(self.device)
#                 target_batch_expanded = target_batch.repeat_interleave(self.num_samples, dim=0).to(self.device)
#                 noise_tensor = torch.normal(mean=0.0, std=self.noise, size=zeta_batch_expanded.shape, device=self.device)
#                 xi_batch = zeta_batch_expanded + noise_tensor
#                 predictions = self.model(xi_batch)
#                 ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
#                 loss_values = ce_loss_fn(predictions, target_batch_expanded)
#                 norm_diffs = torch.norm(xi_batch - zeta_batch_expanded, p=2, dim=1)
#                 inner_terms = (loss_values - self.lambda_param * norm_diffs) / (self.lambda_param * self.epsilon)
#                 exp_terms = torch.exp(inner_terms)
#                 exp_terms_reshaped = exp_terms.view(batch_size, self.num_samples)
#                 inner_expectations = torch.mean(exp_terms_reshaped, dim=1)
#                 batch_dro_loss = torch.log(inner_expectations).mean()
#                 final_loss = (self.lambda_param * self.epsilon) * batch_dro_loss
#                 final_loss.backward()
#                 optimizer.step()
#                 epoch_loss += final_loss.item()
#                 pbar.set_postfix(loss=final_loss.item())
#             avg_epoch_loss = epoch_loss / len(dataloader)
#             loss_history.append(avg_epoch_loss)
#         return loss_history

class SinkhornDROLogisticRGO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 20, num_samples: int = 10, max_iter: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.rgo_inner_lr, self.rgo_inner_steps = rgo_inner_lr, rgo_inner_steps
        self.num_samples, self.max_iter, self.learning_rate, self.batch_size = num_samples, max_iter, learning_rate, batch_size
        self.rgo_vectorized_max_trials = 100 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _rgo_sampler_vectorized(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor, current_model_state: nn.Module, num_samples_to_generate: int, epoch: int) -> torch.Tensor:
        batch_size = x_original_batch.size(0)
        x_orig_detached_batch = x_original_batch.detach()
        x_pert_batch = x_orig_detached_batch.clone()
        lr_inner = self.rgo_inner_lr
        inner_steps = int(min(5, self.rgo_inner_steps * (epoch + 1) / self.max_iter))
        for _ in range(inner_steps):
            x_pert_batch.requires_grad_(True)
            per_sample_losses = self._get_model_loss_value_batched(x_pert_batch, y_original_batch, current_model_state)
            per_sample_grads, = torch.autograd.grad(outputs=per_sample_losses, inputs=x_pert_batch, grad_outputs=torch.ones_like(per_sample_losses))
            x_pert_batch = x_pert_batch.detach()
            grad_total = -per_sample_grads / self.lambda_param + 2 * (x_pert_batch - x_orig_detached_batch)
            x_pert_batch -= lr_inner * grad_total
        x_opt_star_batch = x_pert_batch
        var_rgo = self.epsilon
        if var_rgo <= 1e-12: return x_opt_star_batch.repeat_interleave(num_samples_to_generate, dim=0)
        std_rgo = math.sqrt(var_rgo)
        f_model_loss_opt_star = self._get_model_loss_value_batched(x_opt_star_batch, y_original_batch, current_model_state)
        norm_sq_opt_star = torch.sum((x_opt_star_batch - x_orig_detached_batch) ** 2, dim=1)
        f_L_xi_opt_star = (-f_model_loss_opt_star / (self.lambda_param * self.epsilon)) + (norm_sq_opt_star / self.epsilon)
        x_opt_star_3d, x_original_3d, f_L_xi_opt_star_3d = x_opt_star_batch.unsqueeze(1), x_orig_detached_batch.unsqueeze(1), f_L_xi_opt_star.unsqueeze(1)
        final_accepted_perturbations = torch.zeros((batch_size, num_samples_to_generate, self.input_dim), device=self.device)
        active_flags = torch.ones((batch_size, num_samples_to_generate), dtype=torch.bool, device=self.device)
        for _ in range(self.rgo_vectorized_max_trials):
            if not active_flags.any(): break
            pert_proposals = torch.randn_like(final_accepted_perturbations) * std_rgo
            x_candidates = x_opt_star_3d + pert_proposals
            x_candidates_flat = x_candidates.view(-1, self.input_dim)
            y_repeated = y_original_batch.repeat_interleave(num_samples_to_generate, dim=0)
            f_model_loss_candidates = self._get_model_loss_value_batched(x_candidates_flat, y_repeated, current_model_state).view(batch_size, num_samples_to_generate)
            norm_sq_candidates = torch.sum((x_candidates - x_original_3d) ** 2, dim=2)
            f_L_xi_candidates = (-f_model_loss_candidates / (self.lambda_param * self.epsilon)) + (norm_sq_candidates / self.epsilon)
            diff_cand_opt_norm_sq = torch.sum(pert_proposals**2, dim=2)
            exponent_term3 = diff_cand_opt_norm_sq / (2 * var_rgo)
            acceptance_probs = torch.exp(torch.clamp(-f_L_xi_candidates + f_L_xi_opt_star_3d + exponent_term3, max=10))
            newly_accepted_mask = (torch.rand_like(acceptance_probs) < acceptance_probs) & active_flags
            final_accepted_perturbations[newly_accepted_mask] = pert_proposals[newly_accepted_mask]
            active_flags[newly_accepted_mask] = False
        return (x_opt_star_3d + final_accepted_perturbations).view(-1, self.input_dim)
    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_iter):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Training RGO Epoch {epoch+1}/{self.max_iter}")
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_rgo_batch = self._rgo_sampler_vectorized(x_original_batch_dev, y_original_batch_dev, self.model, self.num_samples, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_rgo_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                comparable_loss = self._compute_dual_loss(x_original_batch, y_original_batch, self.lambda_param, self.epsilon)
                epoch_loss_record += comparable_loss
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss)
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
        return loss_history
    
class SinkhornDROLD(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_iter: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.num_samples, self.max_iter, self.learning_rate, self.batch_size = num_samples, max_iter, learning_rate, batch_size
        self.rgo_vectorized_max_trials = 100 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _LD_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module) -> torch.Tensor:
        batch_size = x_orig.size(0)
        x_orig = x_orig.requires_grad_(True)
        predictions = model(x_orig)
        loss = nn.CrossEntropyLoss(reduction='sum')(predictions, y_orig)
        grad = torch.autograd.grad(loss, x_orig, retain_graph=True)[0]
        mean = x_orig + grad * self.epsilon / 2
        std_dev = torch.sqrt(torch.tensor(self.epsilon, device=self.device))
        mean_expanded = mean.unsqueeze(1).expand(batch_size, self.num_samples, self.input_dim)
        noise = std_dev * torch.randn(batch_size, self.num_samples, self.input_dim, device=self.device)
        X0_torch = mean_expanded + noise
        # Reshape to [batch_size * num_samples, input_dim]
        X0_torch = X0_torch.reshape(-1, self.input_dim)
        return X0_torch
    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_iter):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Training RGO Epoch {epoch+1}/{self.max_iter}")
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_rgo_batch = self._LD_sampler(x_original_batch_dev, y_original_batch_dev, self.model)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_rgo_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                comparable_loss = self._compute_dual_loss(x_original_batch, y_original_batch, self.lambda_param, self.epsilon)
                epoch_loss_record += comparable_loss
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss)
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
        return loss_history


# --- 4. Training and Evaluation Functions ---
def train_saa(model, train_loader, epochs=30, learning_rate=0.001):
    model.to(DEVICE).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, labels in pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
    return loss_history

def evaluate_model(model_object, test_features_np, test_labels_np, attack_fn, epsilon_list):
    pytorch_model = model_object.model.to(DEVICE).eval()
    results = {0.0: model_object.score(test_features_np, test_labels_np) * 100}
    print(f"Accuracy on clean test set: {results[0.0]:.2f}%")
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_features_np), torch.from_numpy(test_labels_np)), batch_size=128)
    for epsilon in epsilon_list:
        correct, total = 0, 0
        for features, labels in tqdm(test_loader, desc=f"Evaluating (epsilon={epsilon:.4f})"):
            perturbed_features = attack_fn(model=pytorch_model, features=features, labels=labels, epsilon=epsilon, alpha=epsilon / 8, num_iter=10)
            with torch.no_grad():
                outputs = pytorch_model(perturbed_features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(DEVICE)).sum().item()
        results[epsilon] = 100 * correct / total
        print(f"Accuracy under PGD attack (epsilon={epsilon:.4f}): {results[epsilon]:.2f}%")
    return results


# --- 5. Plotting Functions ---
def plot_training_loss(loss_dict):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    for model_name, loss_history in loss_dict.items():
        if loss_history:
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='--', label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend(title="Model")
    plt.grid(True, which='both', linestyle='-')
    plt.savefig("training_loss_comparison.png")
    print("\nTraining loss plot saved as 'training_loss_comparison.png'")

def plot_robustness_results(results_dict, perturbation_levels, avg_feature_norm):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    for model_name, results in results_dict.items():
        y_values = [100.0 - results.get(p_level * avg_feature_norm, results.get(0.0, 0.0)) for p_level in perturbation_levels]
        plt.plot(perturbation_levels, y_values, marker='o', linestyle='--', label=model_name)
    plt.xlabel("Level of Perturbations / Norm of Data")
    plt.ylabel("Mis-classification Rate (%)")
    plt.title("Model Robustness under PGD Attack on CIFAR-10 Features")
    plt.legend(title="Model")
    plt.grid(True, which='both', linestyle='-')
    plt.ylim(bottom=0)
    plt.xticks(perturbation_levels)
    plt.savefig("robustness_comparison_plot.png")
    print("\nRobustness plot saved as 'robustness_comparison_plot.png'")


# --- 6. Main Execution Flow ---
if __name__ == '__main__':
    print("--- Loading Pre-extracted Features ---")
    file_path = "cifar10_resnet50_features.pth"
    SEED = 2022
    torch.manual_seed(SEED)
    
    if not os.path.exists(file_path):
        print(f"Error: Feature file '{file_path}' not found.")
    else:
        data = torch.load(file_path)
        train_features, train_labels = data["train_features"], data["train_labels"]
        test_features, test_labels = data["test_features"], data["test_labels"]
        
        train_features_np, train_labels_np = train_features.numpy(), train_labels.numpy()
        test_features_np, test_labels_np = test_features.numpy(), test_labels.numpy()
        
        INPUT_DIM, NUM_CLASSES = train_features_np.shape[1], 10
        DRO_BATCH_SIZE = 128
        lam, eps, itr = 10, 0.01, 30
        
        all_training_losses = {}

        print("\n--- Training SinkhornDROLD Model ---")
        LD_dro_model = SinkhornDROLD(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
        LD_loss_history = LD_dro_model.fit(train_features_np, train_labels_np)
        all_training_losses["LD"] = LD_loss_history

        print("\n--- Training SAA Model ---")
        saa_model_wrapper = BaseLinearDRO(INPUT_DIM, NUM_CLASSES, True)
        saa_model_wrapper.model = LogisticRegression(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        feature_train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=128, shuffle=True)
        saa_loss_history = train_saa(saa_model_wrapper.model, feature_train_loader, epochs=itr)
        #all_training_losses["SAA"] = saa_loss_history

        print("\n--- Training SinkhornBaseDRO (Vectorized) ---")
        base_dro_model = SinkhornBaseDRO(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
        base_loss_history = base_dro_model.fit(train_features_np, train_labels_np)
        all_training_losses["Base"] = base_loss_history


        print("\n--- Training SinkhornDROLogisticRGO Model ---")
        rgo_dro_model = SinkhornDROLogisticRGO(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_iter=10, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
        rgo_loss_history = rgo_dro_model.fit(train_features_np, train_labels_np)
        all_training_losses["RGO"] = rgo_loss_history
        
        plot_training_loss(all_training_losses)

        print("\n--- Evaluating Models ---")
        avg_feature_norm = np.mean(np.linalg.norm(test_features_np, axis=1))
        print(f"Average L2 norm of test set features: {avg_feature_norm:.2f}")
        perturbation_levels = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        epsilon_values = [level * avg_feature_norm for level in perturbation_levels]
        
        all_results = {}
        models_to_evaluate = { "SAA": saa_model_wrapper, "Base": base_dro_model, "LD": LD_dro_model, "RGO": rgo_dro_model }

        for name, model_obj in models_to_evaluate.items():
            print(f"\n--- Evaluating {name} ---")
            results = evaluate_model(model_obj, test_features_np, test_labels_np, pgd_attack, epsilon_values[1:])
            all_results[name] = results
            
        plot_robustness_results(all_results, perturbation_levels, avg_feature_norm)