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

# --- 导入 ---
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    print("scikit-learn not found. Please run: pip install scikit-learn")
    accuracy_score = None
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 全局设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def extract_features(data_loader, model, device):
    model.to(device)
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)

# --- 2. 对抗性攻击 (修正版) ---
def pgd_attack(model, features, labels, epsilon, alpha, num_iter):
    """PGD 对抗性攻击 (l2 范数) - 修正版本"""
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
        # 确保归一化稳定
        grad_norm = torch.linalg.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True) + 1e-12
        normalized_grad = grad / grad_norm
        
        reshape_dims = [grad.shape[0]] + [1] * (grad.dim() - 1)
        
        perturbed_features.data = perturbed_features.data + alpha * normalized_grad
        
        perturbation = perturbed_features.data - original_features.data
        
        # 将扰动投影回 L2 球
        pert_norm = torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, keepdim=True)
        factor = epsilon / (pert_norm + 1e-12)
        factor = torch.min(torch.ones_like(factor), factor)
        
        perturbation = perturbation * factor.view(*reshape_dims)
        perturbed_features.data = original_features.data + perturbation

    return perturbed_features.detach()


# --- 3. 模型定义 ---
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
                 lambda_param: float = 1e2, max_itr: int = 100, learning_rate: float = 1e-2,
                 sample_level: int = 6, batch_size: int = 64, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.max_itr, self.learning_rate, self.sample_level, self.batch_size = epsilon, lambda_param, max_itr, learning_rate, sample_level, batch_size
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(input_dim, output_dim=num_classes, bias=fit_intercept).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        loss_history = []
        self.model.train()
        for epoch in range(self.max_itr):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training SinkhornLinearDRO Epoch {epoch+1}/{self.max_itr}", leave=False)
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
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 20, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.rgo_inner_lr, self.rgo_inner_steps = rgo_inner_lr, rgo_inner_steps
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.rgo_vectorized_max_trials = 100 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _rgo_sampler_vectorized(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor, current_model_state: nn.Module, num_samples_to_generate: int, epoch: int) -> torch.Tensor:
        batch_size = x_original_batch.size(0)
        x_orig_detached_batch = x_original_batch.detach()
        x_pert_batch = x_orig_detached_batch.clone()
        lr_inner = self.rgo_inner_lr
        inner_steps = int(min(5, self.rgo_inner_steps * (epoch + 1) / self.max_itr))
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
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training RGO Epoch {epoch+1}/{self.max_itr}", leave=False)
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

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SinkhornDROLogisticRGO_run{run_id}_epoch_{epoch+1}.pth')

        return loss_history
    
class SinkhornDROLD(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
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
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training LD Epoch {epoch+1}/{self.max_itr}", leave=False)
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

class SinkhornDROWFR(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _WFR_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        """
        使用朗之万动力学生成最坏情况样本及其对应的权重。
        
        该方法实现了一个采样器，通过迭代更新一组粒子（样本）来找到最坏情况分布，
        以最大化模型损失，并受 WFR 距离的约束。现在它包括一个重采样步骤，
        以处理权重递减的粒子。
        """
        batch_size = x_orig.shape[0]

        # 均匀初始化所有样本的权重
        weights = torch.ones((batch_size, self.num_samples), dtype=torch.float32, device=self.device) / self.num_samples

        # 克隆并扩展原始数据以进行样本的批处理
        x_clone = x_orig.clone().detach().unsqueeze(1).expand(-1, self.num_samples, -1).contiguous().view(-1, self.input_dim)
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = x_orig.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, self.input_dim)

        # 预计算权重更新规则的常数
        wt_lr = self.inner_lr
        weight_exponent = 1 - self.lambda_param * self.epsilon * wt_lr

        # 根据当前 epoch 动态调整内部步数
        num_steps = int(max(5, self.inner_itr * (epoch + 1) / self.max_itr))
        
        for _ in range(num_steps):
            x_clone.requires_grad_(True)
            
            # 计算损失关于样本的梯度
            loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach()
            
            # 执行一步朗之万动力学
            mean = x_clone + self.inner_lr * (grads - 2 * self.lambda_param * (x_clone - x_original_expanded))
            std_dev = torch.sqrt(torch.tensor(2 * self.inner_lr * self.lambda_param * self.epsilon, device=self.device))
            
            noise = torch.randn_like(mean) * std_dev
            x_clone = mean + noise

            with torch.no_grad():
                current_loss = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated).view(batch_size, -1)
                dist_sq = torch.sum((x_clone - x_original_expanded) ** 2, dim=1).view(batch_size, -1)
                
                energy_term = current_loss - 2 * self.lambda_param * dist_sq

                weights = weights ** weight_exponent * torch.exp(energy_term * wt_lr)

                weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

                low_weight_mask = weights < 1e-4
                rows_with_low_weights = torch.any(low_weight_mask, dim=1)

                if torch.any(rows_with_low_weights):
                    x_reshaped = x_clone.view(batch_size, self.num_samples, -1)

                    max_weight_vals, max_weight_indices = torch.max(weights, dim=1, keepdim=True)
                    highest_weight_point_data = torch.gather(x_reshaped, 1, max_weight_indices.unsqueeze(-1).expand(-1, -1, self.input_dim))

                    low_weights_sum = torch.sum(weights * low_weight_mask, dim=1, keepdim=True)
                    num_low_weights = torch.sum(low_weight_mask, dim=1, keepdim=True, dtype=weights.dtype)
                    avg_weight = (max_weight_vals + low_weights_sum) / (num_low_weights + 1.0 + 1e-9)
                    avg_weight_expanded = avg_weight.expand_as(weights)

                    max_weight_mask = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, max_weight_indices, True)
                    update_mask = (low_weight_mask | max_weight_mask) & rows_with_low_weights.unsqueeze(1)
                    
                    weights = torch.where(update_mask, avg_weight_expanded, weights)

                    x_update_mask = low_weight_mask & rows_with_low_weights.unsqueeze(1)
                    replacement_data = highest_weight_point_data.expand_as(x_reshaped)
                    x_reshaped = torch.where(x_update_mask.unsqueeze(-1), replacement_data, x_reshaped)
                    x_clone = x_reshaped.view(-1, self.input_dim)
                    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)

        return x_clone.detach(), weights.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WFR Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WFR_batch, weights = self._WFR_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_WFR_batch)
                loss_values = nn.CrossEntropyLoss(reduction='none')(predictions_logits_batch, y_repeated_batch)
                optimization_loss = (loss_values.view(-1, self.num_samples) * weights).sum(dim=1).mean()
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SinkhornDROWFR_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history
    
class SinkhornDROWGF(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _WGF_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        x_clone = x_clone.unsqueeze(1).expand(-1, self.num_samples, -1).contiguous().view(-1, self.input_dim)
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = x_orig.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, self.input_dim)
        num_steps = int(max(5, self.inner_itr * (epoch + 1) / self.max_itr))
        for _ in range(num_steps):
            x_clone.requires_grad_(True)
            loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach() # 计算梯度后分离
            
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
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WGF Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WGF_batch = self._WGF_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_WGF_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SinkhornDROWGF_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history
    
class SinkhornDROSVG(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, 
                 lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, 
                 learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, 
                 inner_itr: int = 200, adagrad_hist_decay: float = 0.9, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr, self.adagrad_hist_decay = inner_lr, inner_itr, adagrad_hist_decay
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)

    def _rbf_kernel_batched_torch(self, particles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为一批粒子计算 RBF 核及其梯度。
        """
        B, S, D = particles.shape
        device = particles.device

        # 计算平方成对距离
        sq_dist = torch.cdist(particles, particles, p=2).pow(2)

        # 使用中值启发法确定带宽 h
        median_sq_dist = torch.median(sq_dist.view(B, -1), dim=1, keepdim=True)[0]
        median_sq_dist = median_sq_dist.view(B, 1, 1)
        h_squared = median_sq_dist / (torch.log(torch.tensor(S, dtype=torch.float32, device=device)) + 1e-8) + 1e-2

        # 计算核矩阵 K
        K = torch.exp(-sq_dist / (2 * h_squared))

        # 计算核的梯度
        diff = particles.unsqueeze(2) - particles.unsqueeze(1)
        grad_K_x = -diff / h_squared.unsqueeze(-1) * K.unsqueeze(-1)

        return K, grad_K_x

    def _svg_sampler(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        使用 Stein 变分梯度下降 (SVGD) 生成最坏情况样本。
        """
        if x_original_batch.shape[0] == 0:
            return torch.empty(0, x_original_batch.shape[1], device=x_original_batch.device)
        
        device = self.device
        B, D = x_original_batch.shape
        S = self.num_samples

        # 准备用于批处理的张量
        x_orig_repeated = x_original_batch.unsqueeze(1).repeat(1, S, 1)
        y_repeated = y_original_batch.repeat_interleave(S)

        # 在原始数据点周围初始化粒子
        particles = x_orig_repeated.clone().detach()
        particles += torch.randn_like(particles) * 0.2
        hist_grad = torch.zeros_like(particles)

        for _ in range(self.inner_itr):
            x_tensor = particles.view(B * S, D).clone().requires_grad_(True)
            x_orig_repeated_flat = x_orig_repeated.view(B * S, D)

            # 目标分布的对数似然梯度
            neg_log_likelihood = nn.CrossEntropyLoss(reduction='sum')(model(x_tensor), y_repeated)
            grad_log_py_x, = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_tensor)
            
            # 对数先验（以原始数据为中心的高斯先验）的梯度
            grad_log_px = -2 * self.lambda_param * (x_tensor - x_orig_repeated_flat)

            # 总梯度是后验的梯度: log p(y|x) + log p(x)
            # 上升方向是 grad(loss) - 2*lambda*(x-x_orig)
            total_grad_flat = (grad_log_py_x + grad_log_px) / (self.lambda_param * self.epsilon)
            total_grad = total_grad_flat.view(B, S, D)

            # 计算核及其梯度
            K, grad_K_x = self._rbf_kernel_batched_torch(particles)
            
            # 来自其他粒子的排斥力
            K_grad_prod = torch.bmm(K, total_grad)
            
            # 趋向更高密度的驱动力
            sum_grad_K = torch.sum(grad_K_x, dim=2)

            # 结合力以进行 SVGD 更新
            svg_grad = (K_grad_prod + sum_grad_K) / S

            with torch.no_grad():
                # Adagrad 优化器以实现稳定更新
                hist_grad = self.adagrad_hist_decay * hist_grad + (1 - self.adagrad_hist_decay) * (svg_grad**2)
                adj_grad = svg_grad / (1e-6 + torch.sqrt(hist_grad))
                particles += self.inner_lr * adj_grad

        return particles.view(B * S, D).detach()

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training SVG Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                
                # 使用 SVG 采样器生成扰动样本
                self.model.eval()
                x_svg_batch = self._svg_sampler(x_original_batch_dev, y_original_batch_dev, self.model)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                
                # 使用生成的样本更新模型参数
                self.model.train()
                predictions_logits_batch = self.model(x_svg_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SinkhornDROSVG_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history

class WRM(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, lambda_param: float = 1.0, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.lambda_param = lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.max_itr, self.learning_rate, self.batch_size = max_itr, learning_rate, batch_size 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _sinha_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        num_steps = int(max(5, self.inner_itr * (epoch + 1) / self.max_itr))
        for _ in range(num_steps):
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
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WRM Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WRM_batch = self._sinha_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                self.model.train()
                predictions_logits_batch = self.model(x_WRM_batch)
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
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/WRM_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history

# --- 4. 训练和评估函数 ---
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


# --- 5. 绘图函数 ---
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
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"training_loss_lam={lam}_eps={eps}.pdf")
    print(f"\n训练损失图已保存为 'training_loss_lam={lam}_eps={eps}.pdf'")

def plot_robustness_results(all_runs_results, perturbation_levels, epsilon_values, lam, eps):
    plt.style.use('jz.mplstyle')
    plt.figure(figsize=(3.8, 2.613), dpi=300, tight_layout=True)
    
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
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"robustness_lam={lam}_eps={eps}.pdf")
    print("\n鲁棒性图已保存为 'robustness_comparison_plot.pdf'")

def get_cifar10_dataloaders(batch_size=128):
    """(已替换) 下载并准备 CIFAR-10 数据集的 DataLoader。"""
    # 为训练集转换添加随机数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 保持测试集转换为确定性，以保证评估的一致性
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# --- 6. 主执行流程 ---
if __name__ == '__main__':

    # --- 设置 ---
    resnet_feature_extractor = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    # 从倒数第二层提取特征
    resnet_feature_extractor.fc = nn.Identity()
    
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"模型将保存在 '{CHECKPOINT_DIR}' 目录中。")
    
    # --- 主要超参数 ---
    INPUT_DIM, NUM_CLASSES = 2048, 10 # ResNet-50 倒数第二层的特征维度
    DRO_BATCH_SIZE = 128
    lam_values = [10]
    eps_values = [0.2, 0.02, 0.002]
    itr = 10 # 迭代次数
    
    N_REPEATS = 5 # 重复实验次数
    BASE_SEED = 2022
    
    lr = 5e-3
    for lam in lam_values:
        for eps in eps_values:
            all_runs_training_losses = {"RGO": [], "WGF": [], "Dual": [], "WFR": [], "SVG": []}
            all_runs_eval_results = {"RGO": [], "WGF": [], "SAA": [], "Dual": [], "WRM": [], "WFR": [], "SVG": []}

            for i in range(N_REPEATS):
                run_seed = BASE_SEED + i
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)
                print(f"\n{'='*20} 开始运行 {i+1}/{N_REPEATS} (种子: {run_seed}) {'='*20}")

                # --- 在每次运行时使用数据增强重新提取特征 ---
                print("为本次运行加载数据并提取特征...")
                cifar_train_loader, cifar_test_loader = get_cifar10_dataloaders(batch_size=128)
                train_features, train_labels = extract_features(cifar_train_loader, resnet_feature_extractor, DEVICE)
                test_features, test_labels = extract_features(cifar_test_loader, resnet_feature_extractor, DEVICE)
                
                train_features_np, train_labels_np = train_features.numpy(), train_labels.numpy()
                test_features_np, test_labels_np = test_features.numpy(), test_labels.numpy()
                
                print("\n--- 训练 SinkhornDROSVG 模型 ---")
                SVG_dro_model = SinkhornDROSVG(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, 
                                           lambda_param=lam, epsilon=eps, max_itr=itr, 
                                           learning_rate=lr, batch_size=DRO_BATCH_SIZE, 
                                           num_samples=16, inner_itr=200)
                SVG_loss_history = SVG_dro_model.fit(train_features_np, train_labels_np, 
                                                 checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                all_runs_training_losses["SVG"].append(SVG_loss_history)

                print("\n--- 训练 SinkhornDROWFR 模型 ---")
                WFR_dro_model = SinkhornDROWFR(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_itr=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE, num_samples=16)
                WFR_loss_history = WFR_dro_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                all_runs_training_losses["WFR"].append(WFR_loss_history)

                print("\n--- 训练 SinkhornDROWGF 模型 ---")
                WGF_dro_model = SinkhornDROWGF(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_itr=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE, num_samples=16)
                WGF_loss_history = WGF_dro_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                all_runs_training_losses["WGF"].append(WGF_loss_history)

                print("\n--- 训练 SAA 模型 ---")
                saa_model_wrapper = BaseLinearDRO(INPUT_DIM, NUM_CLASSES, True)
                saa_model_wrapper.model = LogisticRegression(INPUT_DIM, NUM_CLASSES)
                feature_train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=128, shuffle=True)
                saa_loss_history = train_saa(saa_model_wrapper.model, feature_train_loader, epochs=itr, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                
                print("\n--- 训练 SinkhornBaseDRO (Vectorized) ---")
                base_dro_model = SinkhornBaseDRO(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_itr=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE, sample_level=5)
                base_loss_history = base_dro_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                all_runs_training_losses["Dual"].append(base_loss_history)

                print("\n--- 训练 WRM 模型 ---")
                WRM_model = WRM(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, max_itr=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE)
                WRM_loss_history = WRM_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                
                print("\n--- 训练 SinkhornDROLogisticRGO 模型 ---")
                rgo_dro_model = SinkhornDROLogisticRGO(INPUT_DIM, NUM_CLASSES, device=DEVICE.type, lambda_param=lam, epsilon=eps, max_itr=itr, learning_rate=lr, batch_size=DRO_BATCH_SIZE, num_samples=16)
                rgo_loss_history = rgo_dro_model.fit(train_features_np, train_labels_np, checkpoint_dir=CHECKPOINT_DIR, run_id=i)
                all_runs_training_losses["RGO"].append(rgo_loss_history)

                print(f"\n--- 评估运行 {i+1} 的模型 ---")
                avg_feature_norm = np.mean(np.linalg.norm(test_features_np, axis=1))
                perturbation_levels = [0.0, 0.008, 0.016, 0.024, 0.032, 0.04, 0.048, 0.056, 0.064, 0.072, 0.08]
                epsilon_values = [level * avg_feature_norm for level in perturbation_levels]
                models_to_evaluate = {
                    "SAA": saa_model_wrapper, 
                    "Dual": base_dro_model, 
                    "WGF": WGF_dro_model, 
                    "WRM": WRM_model, 
                    "WFR": WFR_dro_model,
                    "SVG": SVG_dro_model,
                    "RGO": rgo_dro_model
                }

                for name, model_obj in models_to_evaluate.items():
                    print(f"\n--- 评估 {name} ---")
                    results = evaluate_model(model_obj, test_features_np, test_labels_np, pgd_attack, epsilon_values[1:])
                    all_runs_eval_results[name].append(results)

            print(f"\n{'='*20} 所有运行已完成 {'='*20}")
            print("生成包含均值和标准差的最终图表...")

            plot_training_loss(all_runs_training_losses, lam, eps)
            plot_robustness_results(all_runs_eval_results, perturbation_levels, epsilon_values, lam, eps)
