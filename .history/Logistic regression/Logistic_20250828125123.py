# -*- coding: utf-8 -*-
"""
This script runs a hyperparameter sweep to compare the convergence speed and
final performance (F1 Score) of three DRO methods.

The experiment iterates over a grid of epsilon and lambda_param values.
For each combination, it runs the training for three models multiple times.
It records the F1 score on a balanced test set and plots the median loss
curve (recorded every 5 epochs) for convergence analysis.

All generated plots are saved into a 'convergence_plots' directory.
A final summary of F1 scores is printed to the console.
"""

# 1. Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings
import math
import os
from typing import Dict, Union, Tuple, Any, List

# 2. Data Generation
def generate_imbalanced_data(n_samples=1000, noise_level=0.01):
    X = np.random.randn(n_samples, 2).astype(np.float32)
    decision_boundary = X[:, 0] - 2 * X[:, 1]
    noise = noise_level * np.random.randn(n_samples)
    y = (decision_boundary + noise > 0).astype(np.float32)
    minority_class_indices = np.where(y == 1)[0]
    majority_class_indices = np.where(y == 0)[0]
    n_minority = int(n_samples / 4 * 0.1)
    n_majority = int(n_samples / 4 * 0.9)
    if len(minority_class_indices) < n_minority or len(majority_class_indices) < n_majority:
        n_minority = min(n_minority, len(minority_class_indices))
        n_majority = min(n_majority, len(majority_class_indices))
    selected_minority = np.random.choice(minority_class_indices, n_minority, replace=False)
    selected_majority = np.random.choice(majority_class_indices, n_majority, replace=False)
    selected_indices = np.concatenate([selected_minority, selected_majority])
    np.random.shuffle(selected_indices)
    return X[selected_indices], y[selected_indices]

def generate_balanced_data(n_samples=1000, noise_level=0.1):
    X = np.random.randn(n_samples, 2).astype(np.float32)
    decision_boundary = X[:, 0] - 2 * X[:, 1]
    noise = noise_level * np.random.randn(n_samples)
    y = (decision_boundary + noise > 0).astype(np.float32)
    return X, y

# 3. Core Modules and Base Classes
class DROError(Exception):
    pass

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class BaseLinearDRO:
    def __init__(self, input_dim: int, fit_intercept: bool):
        self.input_dim = input_dim
        self.fit_intercept = fit_intercept
        self.device = torch.device("cpu")
        self.model: nn.Module
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1: X = X.reshape(-1, self.input_dim)
        if y.ndim == 1: y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise DROError(f"X and y must have the same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")
        if X.shape[1] != self.input_dim:
            raise DROError(f"Expected input_dim={self.input_dim} features for X, got {X.shape[1]}")
        return X, y
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        X_tensor, y_tensor = self._to_tensor(X), self._to_tensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    def _extract_parameters(self) -> Dict[str, Union[np.ndarray, None]]:
        theta = self.model.linear.weight.detach().cpu().numpy().flatten()
        bias_val = self.model.linear.bias.detach().cpu().numpy() if self.model.linear.bias is not None else None
        return {"theta": theta, "bias": bias_val}
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_val, _ = self._validate_inputs(X, np.zeros(X.shape[0]))
        X_tensor = self._to_tensor(X_val)
        self.model.eval()
        with torch.no_grad():
            predictions_logits = self.model(X_tensor).cpu().numpy()
        return predictions_logits
    def score(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        y_pred_logits = self.predict(X)
        y_true_flat = y.flatten()
        pred_labels_flat = (y_pred_logits.flatten() >= 0).astype(int)
        accuracy = accuracy_score(y_true_flat, pred_labels_flat)
        f1 = f1_score(y_true_flat, pred_labels_flat, average='macro', zero_division=0)
        return accuracy, f1

# 4. Model Implementations
class SinkhornLinearDRO(BaseLinearDRO):
    def __init__(self, input_dim: int, fit_intercept: bool = True, epsilon: float = 1e-3,
                 lambda_param: float = 1e2, max_iter: int = 1000, learning_rate: float = 1e-2,
                 num_samples: int = 32, device: str = "cpu"):
        super().__init__(input_dim, fit_intercept)
        self.epsilon, self.lambda_param, self.max_iter, self.learning_rate, self.num_samples = epsilon, lambda_param, max_iter, learning_rate, num_samples
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(input_dim, output_dim=1, bias=fit_intercept).to(self.device)
    def _compute_loss(self, predictions, targets, m, lambda_reg):
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        residuals = criterion(predictions, targets) / lambda_reg
        residual_matrix = residuals.view(m, -1)
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[float]]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y.reshape(-1, 1), batch_size=1)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        loss_history = []
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_losses = []
            for data, target in dataloader:
                optimizer.zero_grad()
                m = self.num_samples
                expanded_data = data.repeat(m, 1)
                noise = torch.randn_like(expanded_data) * math.sqrt(self.epsilon)
                noisy_data = expanded_data + noise
                repeated_target = target.repeat(m, 1)
                predictions = self.model(noisy_data)
                loss = self._compute_loss(predictions, repeated_target, m, lambda_reg)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            if epoch_losses and (epoch + 1) % 5 == 0:
                loss_history.append(np.mean(epoch_losses))
                
        return self._extract_parameters(), loss_history

class SVGD:
    def _svgd_kernel(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N, D = theta.shape
        pairwise_sq_dists = squareform(pdist(theta, 'sqeuclidean'))
        h2 = 0.5 * np.median(pairwise_sq_dists) / np.log(N + 1)
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
    def __init__(self, input_dim: int, fit_intercept: bool = True, epsilon: float = 0.1,
                 lambda_param: float = 1.0, svgd_n_iter: int = 50, svgd_stepsize: float = 1e-2,
                 num_samples: int = 10, max_iter: int = 100, learning_rate: float = 0.01,
                 device: str = "cpu"):
        super().__init__(input_dim, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate = epsilon, lambda_param, num_samples, max_iter, learning_rate
        self.svgd_n_iter = svgd_n_iter
        self.svgd_stepsize = svgd_stepsize
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(self.input_dim, output_dim=1, bias=self.fit_intercept).to(self.device)
        self.svgd_computer = SVGD()
    def _svgd_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, epoch: int, model: nn.Module) -> torch.Tensor:
        def grad_log_prob(x_np: np.ndarray) -> np.ndarray:
            grad_prior = -2.0 / self.epsilon * (x_np - x_orig.cpu().numpy())
            x_torch = self._to_tensor(x_np).requires_grad_(True)
            y_torch = y_orig.repeat(x_torch.shape[0], 1)
            predictions = model(x_torch)
            loss = nn.BCEWithLogitsLoss(reduction='sum')(predictions, y_torch)
            grad_likelihood = torch.autograd.grad(loss, x_torch, retain_graph=True)[0]
            return grad_prior - (grad_likelihood / (self.lambda_param * self.epsilon)).detach().cpu().numpy()
        mean = x_orig
        std_dev = torch.sqrt(torch.tensor(self.epsilon / 2.0, device=self.device))
        X0_torch = mean + std_dev * torch.randn(self.num_samples, self.input_dim, device=self.device)
        final_particles_np = self.svgd_computer.update(X0_torch.cpu().numpy(), grad_log_prob, int(min(5, self.svgd_n_iter*epoch/self.max_iter)), self.svgd_stepsize)
        return self._to_tensor(final_particles_np)
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[float]]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y.reshape(-1, 1), batch_size=1)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_history = []
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_losses = []
            for x_orig, y_orig in dataloader:
                self.model.eval()
                worst_case_samples = self._svgd_sampler(x_orig, y_orig, epoch, self.model)
                self.model.train()
                y_repeated = y_orig.repeat(self.num_samples, 1)
                predictions = self.model(worst_case_samples)
                loss = nn.BCEWithLogitsLoss()(predictions, y_repeated)
                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_theta.step()
                epoch_losses.append(loss.item())
            
            if epoch_losses and (epoch + 1) % 5 == 0:
                loss_history.append(np.mean(epoch_losses))

        return self._extract_parameters(), loss_history

class SinkhornDROLogisticRGO(BaseLinearDRO):
    def __init__(self, input_dim: int, fit_intercept: bool = True, epsilon: float = 0.1,
                 lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 50,
                 num_samples: int = 10, max_iter: int = 100, learning_rate: float = 0.01,
                 rgo_vectorized_max_trials: int = 40, device: str = "cpu"):
        super().__init__(input_dim, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate = epsilon, lambda_param, num_samples, max_iter, learning_rate
        self.rgo_inner_lr, self.rgo_inner_steps, self.rgo_vectorized_max_trials = rgo_inner_lr, rgo_inner_steps, rgo_vectorized_max_trials
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(self.input_dim, output_dim=1, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        predictions_logits_batch = model_instance(x_features_batch)
        return nn.BCEWithLogitsLoss(reduction='none')(predictions_logits_batch, y_target_batch).squeeze(-1)
    def _get_model_loss_value_scalar(self, x_features: torch.Tensor, y_target: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return self._get_model_loss_value_batched(x_features, y_target, model_instance).sum()
    def _rgo_sampler_vectorized(self, x_original_sample_xi: torch.Tensor, y_original_sample: torch.Tensor,
                                current_model_state: nn.Module, num_samples_to_generate: int, epoch: int) -> torch.Tensor:
        x_orig_detached_xi = x_original_sample_xi.detach()
        x_pert = x_orig_detached_xi.clone()
        lr_inner = self.rgo_inner_lr
        for _ in range(int(min(5, self.rgo_inner_steps * epoch/self.max_iter))):
            x_pert.requires_grad_(True)
            f_model_loss = self._get_model_loss_value_scalar(x_pert, y_original_sample, current_model_state)
            grad_f_model, = torch.autograd.grad(f_model_loss, x_pert, retain_graph=False)
            x_pert = x_pert.detach()
            grad_total = -grad_f_model / self.lambda_param + 2 * (x_pert - x_orig_detached_xi)
            x_pert -= lr_inner * grad_total
        x_opt_star = x_pert
        var_rgo = self.epsilon
        if var_rgo <= 1e-12:
            return x_opt_star.repeat(num_samples_to_generate, 1)
        std_rgo = math.sqrt(var_rgo)
        final_accepted_perturbations = torch.zeros((num_samples_to_generate, self.input_dim), device=self.device)
        active_flags = torch.ones(num_samples_to_generate, dtype=torch.bool, device=self.device)
        f_model_loss_opt_star = self._get_model_loss_value_scalar(x_opt_star, y_original_sample, current_model_state)
        norm_sq_opt_star = torch.sum((x_opt_star - x_orig_detached_xi) ** 2)
        f_L_xi_opt_star_scalar = (-f_model_loss_opt_star / (self.lambda_param * self.epsilon)) + (norm_sq_opt_star / self.epsilon)
        for _ in range(self.rgo_vectorized_max_trials):
            if not active_flags.any(): break
            num_active = int(active_flags.sum())
            pert_proposals = torch.randn((num_active, self.input_dim), device=self.device) * std_rgo
            x_candidates = x_opt_star + pert_proposals
            y_repeated_for_batch = y_original_sample.repeat(num_active, 1)
            f_model_loss_candidates_vec = self._get_model_loss_value_batched(x_candidates, y_repeated_for_batch, current_model_state)
            norm_sq_candidates_vec = torch.sum((x_candidates - x_original_sample_xi) ** 2, dim=1)
            f_L_xi_candidates_vec = (-f_model_loss_candidates_vec / (self.lambda_param * self.epsilon)) + (norm_sq_candidates_vec / self.epsilon)
            diff_cand_opt_norm_sq_vec = torch.sum(pert_proposals**2, dim=1)
            exponent_term3_vec = diff_cand_opt_norm_sq_vec / (2 * var_rgo)
            prob_exp_argument_vec = torch.clamp(-f_L_xi_candidates_vec + f_L_xi_opt_star_scalar + exponent_term3_vec, max=10)
            acceptance_probs_vec = torch.exp(prob_exp_argument_vec)
            rand_unif_vec = torch.rand(num_active, device=self.device)
            newly_accepted_mask_local = rand_unif_vec < acceptance_probs_vec
            active_indices = torch.where(active_flags)[0]
            indices_to_update = active_indices[newly_accepted_mask_local]
            if indices_to_update.numel() > 0:
                final_accepted_perturbations.index_copy_(0, indices_to_update, pert_proposals[newly_accepted_mask_local])
                active_flags.index_fill_(0, indices_to_update, False)
        return x_opt_star + final_accepted_perturbations
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[float]]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y.reshape(-1, 1), batch_size=1)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_history = []
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_losses = []
            for x_original_xi_t, y_original_t in dataloader:
                x_rgo_tm_batch = self._rgo_sampler_vectorized(x_original_xi_t, y_original_t, self.model, self.num_samples, epoch)
                if x_rgo_tm_batch.size(0) == 0: continue
                y_t_repeated_M_times = y_original_t.repeat(self.num_samples, 1)
                predictions_logits_tm_batch = self.model(x_rgo_tm_batch)
                average_loss = nn.BCEWithLogitsLoss(reduction='mean')(predictions_logits_tm_batch, y_t_repeated_M_times)
                if self.num_samples > 0:
                    optimizer_theta.zero_grad()
                    average_loss.backward()
                    optimizer_theta.step()
                    epoch_losses.append(average_loss.item())
            
            if epoch_losses and (epoch + 1) % 5 == 0:
                loss_history.append(np.mean(epoch_losses))
                
        return self._extract_parameters(), loss_history

# 5. New Experiment and Plotting Functions
from matplotlib.ticker import ScalarFormatter, NullFormatter

def plot_smooth_loss_comparison(loss_histories_median, epsilon, lambda_param, save_dir):
    """
    Plots the convergence, letting 'jz.mplstyle' control all core styling.
    All manual style overrides, including bolding, have been removed.
    """
    # 1. 应用样式表
    plt.style.use('./jz.mplstyle')

    # 2. 创建画布 (尺寸 figsize 和 dpi 由样式文件自动设定)
    plt.figure() 
    
    styles = {
        'Dual':            {'color': 'dodgerblue', 'marker': 'o'},
        'Transport': {'color': 'g',          'marker': 's'},
        'RGO':             {'color': '#FF8C00',    'marker': '^'}
    }
    
    # 3. 绘制数据点 (线条粗细 linewidth 和标记大小 markersize 由样式文件自动设定)
    for name, history in loss_histories_median.items():
        if len(history) > 0:
            style_dict = styles[name]
            epochs = np.arange(5, 5 * len(history) + 1, 5)

            plt.plot(epochs, history, 
                     color=style_dict['color'],  
                     linestyle='-',
                     label=name)

    # 4. 设置标题和轴标签 (字号 fontsize 和字重 fontweight 由样式文件自动设定)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    title = f'$\\epsilon$={epsilon}, $\\lambda$={lambda_param}'
    plt.title(title)

    # 5. 设置图例 (字号和字重由样式文件自动设定)
    plt.legend(loc='best')
    # legend.get_frame().set_edgecolor('black')
    # legend.get_frame().set_linewidth(0.75)
    
    # 6. 设置坐标轴
    plt.yscale('log')
    ax = plt.gca()
    
    # 7. 设置Y轴刻度
    tick_values = [0.03, 0.05, 0.10, 0.20]
    ax.set_yticks(tick_values)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()

    # 8. 刻度数字字体由样式文件自动设定
    
    # 9. 边框粗细由样式文件自动设定

    # 10. 调整布局并保存 (dpi 由样式文件自动设定)
    plt.tight_layout(pad=0.2)
    filename = f"loss_comparison_eps_{str(epsilon).replace('.', '_')}_lambda_{int(lambda_param)}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

def summarize_f1_scores(f1_results, epsilon_values, lambda_values, output_file="f1_score_summary.txt"):
    lines = []
    lines.append("\n" + "="*60)
    lines.append(" " * 18 + "Final F1 Score Summary")
    lines.append("="*60)
    lines.append(f"{'Epsilon':<10} | {'Lambda':<10} | {'Method':<18} | {'Mean F1':<12} | {'Std F1'}")
    lines.append("-"*60)
    
    for epsilon in epsilon_values:
        for lambda_param in lambda_values:
            for name, scores in f1_results.items():
                score_list = scores.get((epsilon, lambda_param), [])
                if score_list:
                    mean_f1 = np.mean(score_list)
                    std_f1 = np.std(score_list)
                    lines.append(f"{epsilon:<10.4f} | {lambda_param:<10.0f} | {name:<18} | {mean_f1:<12.4f} | {std_f1:.4f}")
            lines.append("-"*60)
    summary_text = "\n".join(lines)
    print(summary_text)
    with open(output_file, "w") as f:
        f.write(summary_text)
    print(f"F1 score summary saved to '{output_file}'")

def run_hyperparameter_comparison():
    # --- Experiment Grid ---
    epsilon_values = [0.0001, 0.001, 0.01]
    lambda_values = [10, 100, 1000]
    n_repeats = 5
    
    # --- Setup ---
    model_names = ['Dual', 'Transport', 'RGO']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plot_dir = "convergence_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\nRunning hyperparameter comparison on device: '{device}'")
    print(f"Plots will be saved to '{plot_dir}/'")

    f1_results = {name: {} for name in model_names}

    for epsilon in epsilon_values:
        for lambda_param in lambda_values:
            print(f"\n{'='*20} Testing eps={epsilon}, lambda={lambda_param} {'='*20}")
            
            all_loss_histories = {name: [] for name in model_names}
            for name in model_names:
                f1_results[name][(epsilon, lambda_param)] = []

            for i in range(n_repeats):
                print(f"--- Repetition {i+1}/{n_repeats} ---")
                X_train, y_train = generate_imbalanced_data(n_samples=400, noise_level=0)
                X_test, y_test = generate_balanced_data(n_samples=2000, noise_level=0.3)
                
                common_params = {'max_iter': 50, 'learning_rate': 0.01, 'device': device, 'fit_intercept': True}
                dro_params = {'epsilon': epsilon, 'lambda_param': lambda_param, 'num_samples': 8}
                models_to_run = {
                    'Dual': lambda: SinkhornLinearDRO(input_dim=2, **dro_params, **common_params),
                    'Transport': lambda: SinkhornDROLogisticSVGD(input_dim=2, svgd_n_iter=50, **dro_params, **common_params),
                    'RGO': lambda: SinkhornDROLogisticRGO(input_dim=2, rgo_inner_steps=20, **dro_params, **common_params)
                }
                
                for name, model_constructor in models_to_run.items():
                    try:
                        model = model_constructor()
                        _, loss_history = model.fit(X_train, y_train)
                        all_loss_histories[name].append(loss_history)
                        
                        _, f1 = model.score(X_test, y_test)
                        f1_results[name][(epsilon, lambda_param)].append(f1)
                        print(f"F1 Score for {name}: {f1:.4f}")
                    except Exception as e:
                        print(f"ERROR training {name}: {e}")
                        all_loss_histories[name].append([])
            
            median_histories = {}
            for name, histories in all_loss_histories.items():
                valid_histories = [h for h in histories if h]
                if not valid_histories:
                    median_histories[name] = []
                    continue
                min_len = min(len(h) for h in valid_histories)
                truncated_histories = [h[:min_len] for h in valid_histories]
                median_curve = np.median(np.array(truncated_histories), axis=0)
                median_histories[name] = median_curve
            
            plot_smooth_loss_comparison(median_histories, epsilon, lambda_param, plot_dir)

    summarize_f1_scores(f1_results, epsilon_values, lambda_values)
    print(f"\n{'='*25} Experiment Complete {'='*25}")

if __name__ == "__main__":
    run_hyperparameter_comparison()