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
        # Ensure normalization is stable
        grad_norm = torch.linalg.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True) + 1e-12
        normalized_grad = grad / grad_norm
        
        # In the original code, the view was (-1, 1), which only works for 2D tensors.
        # This is more general for N-D tensors like images.
        reshape_dims = [grad.shape[0]] + [1] * (grad.dim() - 1)
        
        perturbed_features.data = perturbed_features.data + alpha * normalized_grad
        
        perturbation = perturbed_features.data - original_features.data
        
        # Project perturbation back to L2 ball
        pert_norm = torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, keepdim=True)
        factor = epsilon / (pert_norm + 1e-12)
        factor = torch.min(torch.ones_like(factor), factor)
        
        perturbation = perturbation * factor.view(*reshape_dims)
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
        self.model: nn.Module
        self.sample_level = sample_level
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor: return torch.as_tensor(data, dtype=torch.float32)
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1: X = X.reshape(-1, self.input_dim)
        if y.ndim > 1: y = y.flatten()
        if X.shape[0] != y.shape[0]: raise DROError(f"Shapes mismatch: X {X.shape[0]}, y {y.shape[0]}")
        if X.shape[1] != self.input_dim: raise DROError(f"Input dim mismatch: expected {self.input_dim}, got {X.shape[1]}")
        return X, y
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        dataset = TensorDataset(self._to_tensor(X), torch.as_tensor(y, dtype=torch.long))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_val, _ = self._validate_inputs(X, np.zeros(X.shape[0]))
        self.model.eval()
        with torch.no_grad():
            model_device = next(self.model.parameters()).device
            inputs = self._to_tensor(X_val).to(model_device)
            return self.model(inputs).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if accuracy_score is None: return -1.0
        return accuracy_score(y.flatten(), np.argmax(self.predict(X), axis=1))
    
    def _compute_loss(self, predictions, targets, m, lambda_reg):
        criterion = nn.CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg
    
    def _compute_dual_loss(self, data, targets, lam, epsilon):
        m = 2 ** 6
        
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

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        loss_history = []
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training SinkhornLinearDRO Epoch {epoch+1}/{self.max_iter}", leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
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
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SinkhornBaseDRO_run{run_id}_epoch_{epoch+1}.pth')

        return loss_history


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
    
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_iter):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training RGO Epoch {epoch+1}/{self.max_iter}", leave=False)
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
                comparable_loss = self._compute_dual_loss(x_original_batch_dev, y_original_batch_dev, self.lambda_param, self.epsilon)
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SinkhornDROLogisticRGO_run{run_id}_epoch_{epoch+1}.pth')

        return loss_history
    
class SinkhornDROLD(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_iter: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.num_samples, self.max_iter, self.learning_rate, self.batch_size = num_samples, max_iter, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _LD_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module) -> torch.Tensor:
        batch_size = x_orig.size(0)
        x_orig_grad = x_orig.clone().detach().requires_grad_(True)
        predictions = model(x_orig_grad)
        loss = nn.CrossEntropyLoss(reduction='sum')(predictions, y_orig)
        grad = torch.autograd.grad(loss, x_orig_grad, retain_graph=True)[0]
        mean = x_orig + grad  / (2 * self.lambda_param)
        std_dev = torch.sqrt(torch.tensor(self.epsilon, device=self.device))
        mean_expanded = mean.unsqueeze(1).expand(batch_size, self.num_samples, self.input_dim)
        noise = std_dev * torch.randn(batch_size, self.num_samples, self.input_dim, device=self.device)
        X0_torch = mean_expanded + noise
        X0_torch = X0_torch.reshape(-1, self.input_dim)
        return X0_torch
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_iter):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training LD Epoch {epoch+1}/{self.max_iter}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_ld_batch = self._LD_sampler(x_original_batch_dev, y_original_batch_dev, self.model)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_ld_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                comparable_loss = self._compute_dual_loss(x_original_batch_dev, y_original_batch_dev, self.lambda_param, self.epsilon)
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SinkhornDROLD_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history

class SinkhornDROMultiLD(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_iter: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.001, inner_itr: int = 100, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.num_samples, self.max_iter, self.learning_rate, self.batch_size = num_samples, max_iter, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _MultiLD_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        x_clone = x_clone.unsqueeze(1).expand(-1, self.num_samples, -1).contiguous().view(-1, self.input_dim)
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = x_orig.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, self.input_dim)
        
        for _ in range(self.inner_itr):
            x_clone.requires_grad_(True)
            loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach() # Detach after grad calculation
            
            mean = x_clone + self.inner_lr * (grads - 2*self.lambda_param*(x_clone - x_original_expanded))
            std_dev = torch.sqrt(torch.tensor(2*self.inner_lr*self.lambda_param*self.epsilon, device=self.device))
            
            noise = torch.randn_like(mean) * std_dev
            x_clone = mean + noise

        return x_clone.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_iter):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training MultiLD Epoch {epoch+1}/{self.max_iter}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_multild_batch = self._MultiLD_sampler(x_original_batch_dev, y_original_batch_dev, self.model)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_multild_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                comparable_loss = self._compute_dual_loss(x_original_batch_dev, y_original_batch_dev, self.lambda_param, self.epsilon)
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SinkhornDROMultiLD_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history
    
class WDRO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, lambda_param: float = 1.0, max_iter: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.001, inner_itr: int = 100, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.lambda_param = lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.max_iter, self.learning_rate, self.batch_size = max_iter, learning_rate, batch_size 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _sinha_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        for _ in range(self.inner_itr):
            loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_orig)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach() + self.inner_lr * (grads - 2*self.lambda_param*(x_clone - x_orig))
            x_clone.requires_grad_(True)
        return x_clone.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_iter):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WDRO Epoch {epoch+1}/{self.max_iter}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_wdro_batch = self._sinha_sampler(x_original_batch_dev, y_original_batch_dev, self.model)
                self.model.train()
                predictions_logits_batch = self.model(x_wdro_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_original_batch_dev)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                with torch.no_grad():
                    comparable_loss = nn.CrossEntropyLoss()(self.model(x_original_batch_dev), y_original_batch_dev)
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/WDRO_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history

# --- 4. Training and Evaluation Functions ---
def train_saa(model, train_loader, epochs=30, learning_rate=0.001, checkpoint_dir: str = 'checkpoints', run_id: int = 0):
    model.to(DEVICE).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Run {run_id+1} SAA Epoch {epoch+1}/{epochs}", leave=False)
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
        
        torch.save(model.state_dict(), f'{checkpoint_dir}/SAA_run{run_id}_epoch_{epoch+1}.pth')
        
    return loss_history

def evaluate_model(model_object, test_features_np, test_labels_np, attack_fn, epsilon_list):
    pytorch_model = model_object.model.eval()
    
    clean_accuracy = model_object.score(test_features_np, test_labels_np) * 100
    results = {0.0: clean_accuracy}
    print(f"Accuracy on clean test set: {results[0.0]:.2f}%")

    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_features_np).float(), torch.from_numpy(test_labels_np).long()), batch_size=128)
    
    for epsilon in epsilon_list:
        if epsilon == 0.0: continue
        correct, total = 0, 0
        for features, labels in tqdm(test_loader, desc=f"Evaluating (epsilon={epsilon:.4f})", leave=False):
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
def plot_training_loss(all_runs_losses, lam, eps):
    plt.style.use('jz.mplstyle')
    plt.figure()
    
    for model_name, loss_runs in all_runs_losses.items():
        if not loss_runs:
            continue
        
        loss_array = np.array(loss_runs)
        epochs = np.arange(1, loss_array.shape[1] + 1)
        mean_loss = np.mean(loss_array, axis=0)
        std_loss = np.std(loss_array, axis=0)
        
        line, = plt.plot(epochs, mean_loss, label=model_name)
        plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=line.get_color(), alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"training_loss_lam={lam}_eps={eps}.png")
    print(f"\nTraining loss plot saved as 'training_loss_lam={lam}_eps={eps}.png'")

def plot_robustness_results(all_runs_results, perturbation_levels, epsilon_values, lam, eps):
    plt.style.use('jz.mplstyle')
    plt.figure()
    
    for model_name, results_runs in all_runs_results.items():
        if not results_runs:
            continue
            
        errors_matrix = []
        query_keys = [0.0] + epsilon_values[1:]

        for single_run_result in results_runs:
            ordered_errors = [100.0 - single_run_result.get(key, 0.0) for key in query_keys]
            errors_matrix.append(ordered_errors)
        
        errors_array = np.array(errors_matrix)
        mean_errors = np.mean(errors_array, axis=0)
        std_errors = np.std(errors_array, axis=0)
        
        line, = plt.plot(perturbation_levels, mean_errors, label=model_name)
        plt.fill_between(
            perturbation_levels,
            mean_errors - std_errors,
            mean_errors + std_errors,
            color=line.get_color(),
            alpha=0.25
        )

    plt.xlabel("perturbation $\Delta$")
    plt.ylabel("test error (%)")
    plt.title("Test error on CIFAR-10 Features")
    plt.legend(loc='best')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"robustness_lam={lam}_eps={eps}.png")
    print("\nRobustness plot saved as 'robustness_comparison_plot.png'")


# --- 6. Main Execution Flow ---
if __name__ == '__main__':

    # --- Setup ---
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Models will be saved in the '{CHECKPOINT_DIR}' directory.")
    
    print("\n--- Loading Pre-extracted Features ---")
    file_path = "cifar10_resnet50_features.pth"
    
    if not os.path.exists(file_path):
        print(f"Error: Feature file '{file_path}' not found.")
        print("Please run the feature extraction script first or download the features.")
    else:
        data = torch.load(file_path, map_location=DEVICE)
        train_features, train_labels = data["train_features"], data["train_labels"]
        test_features, test_labels = data["test_features"], data["test_labels"]
        
        train_features_np, train_labels_np = train_features.cpu().numpy(), train_labels.cpu().numpy()
        test_features_np, test_labels_np = test_features.cpu().numpy(), test_labels.cpu().numpy()
        
        INPUT_DIM, NUM_CLASSES = train_features_np.shape[1], 10
        DRO_BATCH_SIZE = 128
        lam_values = [1.0, 10.0]
        eps_values = [0.01, 0.001]
        itr = 10 # Hyperparameters
        
        N_REPEATS = 1
        BASE_SEED = 2022
        
        lr = 5e-3
        for lam in lam_values:
            for eps in eps_values:
                all_runs_training_losses = {"MultiLD": [], "Dual": [], "RGO": []}
                all_runs_eval_results = {"MultiLD": [], "SAA": [], "Dual": [], "WDRO": [], "RGO": []}
                for i in range(N_REPEATS):
                    run_seed = BASE_SEED + i
                    torch.manual_seed(run_seed)
                    np.random.seed(run_seed)
                    print(f"\n{'='*20} STARTING RUN {i+1}/{N_REPEATS} (Seed: {run_seed}) {'='*20}")

                    print("\n--- Training SinkhornDROMultiLD Model ---")
                    MultiLD_dro_model = SinkhornDROMultiLD(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE, num_samples=16)
                    MultiLD_loss_history = MultiLD_dro_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                    all_runs_training_losses["MultiLD"].append(MultiLD_loss_history)

                    print("\n--- Training SAA Model ---")
                    saa_model_wrapper = BaseLinearDRO(INPUT_DIM, NUM_CLASSES, True)
                    saa_model_wrapper.model = LogisticRegression(INPUT_DIM, NUM_CLASSES)
                    feature_train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=128, shuffle=True)
                    saa_loss_history = train_saa(saa_model_wrapper.model, feature_train_loader, epochs=itr, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                    
                    print("\n--- Training SinkhornBaseDRO (Vectorized) ---")
                    base_dro_model = SinkhornBaseDRO(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE, sample_level=5)
                    base_loss_history = base_dro_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                    all_runs_training_losses["Dual"].append(base_loss_history)

                    print("\n--- Training WDRO Model ---")
                    WDRO_model = WDRO(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, max_iter=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE)
                    WDRO_loss_history = WDRO_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                    
                    print("\n--- Training SinkhornDROLogisticRGO Model ---")
                    rgo_dro_model = SinkhornDROLogisticRGO(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE, num_samples=16)
                    rgo_loss_history = rgo_dro_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                    all_runs_training_losses["RGO"].append(rgo_loss_history)

                    print("\n--- Evaluating Models for Run {i+1} ---")
                    avg_feature_norm = np.mean(np.linalg.norm(test_features_np, axis=1))
                    perturbation_levels = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
                    epsilon_values = [level * avg_feature_norm for level in perturbation_levels]
                    
                    models_to_evaluate = {"SAA": saa_model_wrapper, "Dual": base_dro_model, "MultiLD": MultiLD_dro_model, "WDRO": WDRO_model, "RGO": rgo_dro_model}

                    with torch.no_grad():
                        for name, model_obj in models_to_evaluate.items():
                            print(f"\n--- Evaluating {name} ---")
                            results = evaluate_model(model_obj, test_features_np, test_labels_np, pgd_attack, epsilon_values[1:])
                            all_runs_eval_results[name].append(results)

                print(f"\n{'='*20} ALL RUNS COMPLETED {'='*20}")
                print("Generating final plots with mean and standard deviation...")

                plot_training_loss(all_runs_training_losses, lam, eps)
                plot_robustness_results(all_runs_eval_results, perturbation_levels, epsilon_values, lam, eps)

