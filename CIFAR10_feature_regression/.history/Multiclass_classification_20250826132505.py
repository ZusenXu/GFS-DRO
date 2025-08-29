import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import math
from typing import Tuple, Dict, List, Union

# --- Imports from your provided code ---
from scipy.spatial.distance import pdist, squareform
# Note: scikit-learn is a common dependency for these metrics.
# If not installed, run: pip install scikit-learn
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    print("scikit-learn not found. Accuracy will not be calculated in the DRO classes.")
    print("Please run: pip install scikit-learn")
    accuracy_score = None

# --- NEW: Imports for plotting ---
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. Global Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. Data Loading and Feature Extraction (Unchanged) ---

def get_cifar10_dataloaders(batch_size=128):
    """Downloads and prepares the DataLoader for the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def extract_features(data_loader, device):
    """Extracts features using a pre-trained ResNet-18 model."""
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_features), torch.cat(all_labels)

# --- 3. Adversarial Attack ---

# MODIFIED: Now uses the custom loss function
def pgd_attack(model, features, labels, epsilon, alpha, num_iter):
    """PGD Adversarial Attack (l2 norm)"""
    perturbed_features = features.clone().detach().to(DEVICE)
    perturbed_features.requires_grad = True
    original_features = features.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        model.zero_grad()
        outputs = model(perturbed_features)
        loss = criterion(outputs, labels)
        loss.backward()

        grad = perturbed_features.grad.detach()
        perturbed_features.data = perturbed_features.data + alpha * grad.sign()
        
        perturbation = perturbed_features.data - original_features.data
        norm = torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1)
        factor = epsilon / (norm + 1e-12)
        factor = torch.min(torch.ones_like(norm), factor)
        
        perturbation = perturbation * factor.view(-1, 1)
        perturbed_features.data = original_features.data + perturbation

    return perturbed_features.detach()

# --- 4. Model ---


# Original Logistic Regression for SAA
class LogisticRegression(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

# --- Your Provided Code (Adapted for Multi-Class and Batching) ---

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

    def _extract_parameters(self) -> Dict[str, Union[np.ndarray, None]]:
        theta = self.model.linear.weight.detach().cpu().numpy()
        bias_val = self.model.linear.bias.detach().cpu().numpy() if self.model.linear.bias is not None else None
        return {"theta": theta, "bias": bias_val}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_val, _ = self._validate_inputs(X, np.zeros(X.shape[0]))
        X_tensor = self._to_tensor(X_val)
        self.model.eval()
        with torch.no_grad():
            predictions_logits = self.model(X_tensor).cpu().numpy()
        return predictions_logits

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if accuracy_score is None: return -1.0
        y_pred_logits = self.predict(X)
        y_true_flat = y.flatten()
        pred_labels_flat = np.argmax(y_pred_logits, axis=1)
        accuracy = accuracy_score(y_true_flat, pred_labels_flat)
        return accuracy
    
    def _compute_loss(self, predictions, targets, m, lambda_reg):
        # MODIFIED: Use the custom loss function
        criterion = nn.CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg

class SinkhornLinearDRO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 1e-3,
                 lambda_param: float = 1e2, max_iter: int = 100, learning_rate: float = 1e-2,
                 num_samples: int = 32, batch_size: int = 64, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.max_iter, self.learning_rate, self.num_samples, self.batch_size = epsilon, lambda_param, max_iter, learning_rate, num_samples, batch_size
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(input_dim, output_dim=num_classes, bias=fit_intercept).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        
        self.model.train()
        for epoch in range(self.max_iter):
            pbar = tqdm(dataloader, desc=f"Training SinkhornLinearDRO Epoch {epoch+1}/{self.max_iter}")
            for data, target in pbar:
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
                 batch_size: int = 64, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate, self.batch_size = epsilon, lambda_param, num_samples, max_iter, learning_rate, batch_size
        self.svgd_n_iter = svgd_n_iter
        self.svgd_stepsize = svgd_stepsize
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
        self.svgd_computer = SVGD()

    def _svgd_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch: int) -> torch.Tensor:
        def grad_log_prob(x_np: np.ndarray) -> np.ndarray:
            grad_prior = -2.0 / self.epsilon * (x_np - x_orig.cpu().numpy())
            x_torch = self._to_tensor(x_np).requires_grad_(True)
            y_torch = y_orig.repeat(x_torch.shape[0])
            predictions = model(x_torch)
            # MODIFIED: Use the custom loss function
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
        lambda_reg = self.lambda_param * self.epsilon
        
        for epoch in range(self.max_iter):
            pbar = tqdm(dataloader, desc=f"Training SVGD Epoch {epoch+1}/{self.max_iter}")
            for x_batch, y_batch in pbar:
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
                
                dro_loss = self._compute_loss(predictions, y_repeated, self.num_samples, lambda_reg)
                criterion = nn.CrossEntropyLoss(reduction='mean')
                loss = criterion(predictions, y_repeated)
                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_theta.step()
                pbar.set_postfix(loss=dro_loss.item())

class SinkhornDROLogisticRGO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1,
                 lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 20,
                 num_samples: int = 10, max_iter: int = 30, learning_rate: float = 0.01,
                 batch_size: int = 64, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate, self.batch_size = epsilon, lambda_param, num_samples, max_iter, learning_rate, batch_size
        self.rgo_inner_lr, self.rgo_inner_steps = rgo_inner_lr, rgo_inner_steps
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)

    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        predictions_logits_batch = model_instance(x_features_batch)
        # MODIFICATION: Changed loss to CrossEntropyLoss for multi-class task
        return nn.CrossEntropyLoss(reduction='none')(predictions_logits_batch, y_target_batch)

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        lambda_reg = self.lambda_param * self.epsilon
        
        for epoch in range(self.max_iter):
            pbar = tqdm(dataloader, desc=f"Training RGO Epoch {epoch+1}/{self.max_iter}")
            for x_original_batch, y_original_batch in pbar:
                all_rgo_samples = []
                for i in range(x_original_batch.size(0)):
                    x_i = x_original_batch[i]
                    y_i = y_original_batch[i]
                    
                    # Generate num_samples for the single (x_i, y_i) pair
                    x_rgo_samples_for_i = self._rgo_sampler_vectorized(
                        x_i, y_i, self.model, self.num_samples, epoch
                    )
                    all_rgo_samples.append(x_rgo_samples_for_i)
                
                # Concatenate all generated samples into a new, larger batch
                # Shape: [batch_size * num_samples, input_dim]
                x_rgo_batch = torch.cat(all_rgo_samples, dim=0)
                
                # Create the corresponding repeated y tensor
                # Shape: [batch_size * num_samples]
                y_repeated_batch = y_original_batch.repeat_interleave(self.num_samples, dim=0)

                self.model.train()
                predictions_logits_batch = self.model(x_rgo_batch)
                loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)

                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_theta.step()
                pbar.set_postfix(loss=loss.item())
class RGO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1,
                 lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 50,
                 num_samples: int = 10, max_iter: int = 100, learning_rate: float = 0.01,
                 rgo_vectorized_max_trials: int = 40, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.num_samples, self.max_iter, self.learning_rate = epsilon, lambda_param, num_samples, max_iter, learning_rate
        self.rgo_inner_lr, self.rgo_inner_steps, self.rgo_vectorized_max_trials = rgo_inner_lr, rgo_inner_steps, rgo_vectorized_max_trials
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        
        # MODIFICATION: Changed output_dim to num_classes for multi-class classification
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)

    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        predictions_logits_batch = model_instance(x_features_batch)
        # MODIFICATION: Changed loss to CrossEntropyLoss for multi-class task
        return nn.CrossEntropyLoss(reduction='none')(predictions_logits_batch, y_target_batch)

    def _get_model_loss_value_scalar(self, x_features: torch.Tensor, y_target: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return self._get_model_loss_value_batched(x_features, y_target, model_instance).sum()

    def _rgo_sampler_vectorized(self, x_original_sample_xi: torch.Tensor, y_original_sample: torch.Tensor,
                                current_model_state: nn.Module, num_samples_to_generate: int, epoch: int) -> torch.Tensor:
        # This is your exact sampler logic, preserved without changes.
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
            # MODIFICATION: Ensure y_repeated has the correct shape for the loss function
            y_repeated_for_batch = y_original_sample.repeat(num_active)
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

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str, interval: int = 10) -> Tuple[Dict[str, np.ndarray], List[float]]:
        # This is your exact fit logic, preserved with minor adaptations.
        X, y = self._validate_inputs(X, y)
        # MODIFICATION: Dataloader now handles 1D long labels correctly
        dataloader = self._create_dataloader(X, y, batch_size=1)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_history = []
        self.model.train()
        print("Starting RGO model training...")
        for epoch in range(1, self.max_iter + 1):
            epoch_loss = 0
            pbar = tqdm(dataloader, desc=f"Training RGO Epoch {epoch}/{self.max_iter}")
            for x_original_xi_t, y_original_t in pbar:
                x_rgo_tm_batch = self._rgo_sampler_vectorized(x_original_xi_t, y_original_t, self.model, self.num_samples, epoch)
                if x_rgo_tm_batch.size(0) == 0: continue
                y_t_repeated_M_times = y_original_t.repeat(self.num_samples)
                predictions_logits_tm_batch = self.model(x_rgo_tm_batch)
                
                # MODIFICATION: Changed loss to CrossEntropyLoss for multi-class task
                average_loss = nn.CrossEntropyLoss(reduction='mean')(predictions_logits_tm_batch, y_t_repeated_M_times)
                
                if self.num_samples > 0:
                    optimizer_theta.zero_grad()
                    average_loss.backward()
                    optimizer_theta.step()
                    loss_history.append(average_loss.item())
                    epoch_loss += average_loss.item()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {epoch} Complete. Average Loss: {avg_epoch_loss:.6f}")
            
            if epoch % interval == 0:
                checkpoint_path = f"{checkpoint_dir}/RGO_model_epoch_{epoch}.ckpt"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        return self._extract_parameters(), loss_history
# --- 5. Training and Evaluation ---

# MODIFIED: Now uses the custom loss function
def train_saa(model, train_loader, epochs=30, learning_rate=0.001):
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, labels in pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

def evaluate_model(model_object, test_features_np, test_labels_np, attack_fn, epsilon_list):
    pytorch_model = model_object.model
    pytorch_model.to(DEVICE)
    pytorch_model.eval()
    results = {}
    clean_accuracy = model_object.score(test_features_np, test_labels_np) * 100
    print(f"Accuracy on clean test set: {clean_accuracy:.2f}%")
    results[0.0] = clean_accuracy
    test_tensor_dataset = TensorDataset(torch.from_numpy(test_features_np), torch.from_numpy(test_labels_np))
    test_loader = DataLoader(test_tensor_dataset, batch_size=128)
    for epsilon in epsilon_list:
        correct, total = 0, 0
        pbar_desc = f"Evaluating (epsilon={epsilon:.4f})"
        for features, labels in tqdm(test_loader, desc=pbar_desc):
            perturbed_features = attack_fn(model=pytorch_model, features=features, labels=labels, epsilon=epsilon, alpha=epsilon / 8, num_iter=10)
            with torch.no_grad():
                outputs = pytorch_model(perturbed_features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(DEVICE)).sum().item()
        perturbed_accuracy = 100 * correct / total
        print(f"Accuracy under PGD attack (epsilon={epsilon:.4f}): {perturbed_accuracy:.2f}%")
        results[epsilon] = perturbed_accuracy
    return results

# --- NEW: Plotting Function ---
def plot_results(results_dict, perturbation_levels):
    """
    Plots the mis-classification rate vs. perturbation level for all models.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    for model_name, results in results_dict.items():
        sorted_epsilons = sorted(results.keys())
        x_values = perturbation_levels
        y_values = [100.0 - results[eps] for eps in sorted_epsilons]
        plt.plot(x_values, y_values, marker='o', linestyle='--', label=model_name)

    plt.xlabel("Level of Perturbations / Norm of Data")
    plt.ylabel("Mis-classification Rate (%)")
    plt.title("Model Robustness under PGD Attack on CIFAR-10 Features")
    plt.legend(title="Model")
    plt.grid(True, which='both', linestyle='-')
    plt.ylim(bottom=0)
    plt.xticks(perturbation_levels)
    plt.savefig("robustness_comparison_plot.png")
    print("\nPlot saved as 'robustness_comparison_plot.png'")
    # plt.show()


# --- 6. Main Execution Flow ---

if __name__ == '__main__':
    print("Loading data and extracting features...")
    cifar_train_loader, cifar_test_loader = get_cifar10_dataloaders()
    train_features, train_labels = extract_features(cifar_train_loader, DEVICE)
    test_features, test_labels = extract_features(cifar_test_loader, DEVICE)
    
    train_features_np, train_labels_np = train_features.numpy(), train_labels.numpy()
    test_features_np, test_labels_np = test_features.numpy(), test_labels.numpy()
    
    INPUT_DIM = train_features_np.shape[1]
    DRO_BATCH_SIZE = 128
    NUM_CLASSES = 10
    lam = 100
    eps = 0.1
    itr = 1
    print("\n--- Training SAA Model ---")
    saa_model_wrapper = BaseLinearDRO(INPUT_DIM, NUM_CLASSES, True)
    saa_model_wrapper.model = LogisticRegression(INPUT_DIM, NUM_CLASSES).to(DEVICE)
    feature_train_dataset = TensorDataset(train_features, train_labels)
    feature_train_loader = DataLoader(feature_train_dataset, batch_size=128, shuffle=True)
    train_saa(saa_model_wrapper.model, feature_train_loader, epochs=1)

    print("\n--- Training SinkhornDROLogisticRGO Model ---")
    rgo_dro_model = SinkhornDROLogisticRGO(INPUT_DIM, NUM_CLASSES, device=DEVICE, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
    rgo_dro_model.fit(train_features_np, train_labels_np)

    print("\n--- Training SinkhornLinearDRO Model ---")
    sl_dro_model = SinkhornLinearDRO(INPUT_DIM, NUM_CLASSES, device=DEVICE, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
    sl_dro_model.fit(train_features_np, train_labels_np)

    print("\n--- Training SinkhornDROLogisticSVGD Model ---")
    svgd_dro_model = SinkhornDROLogisticRGO(INPUT_DIM, NUM_CLASSES, device=DEVICE, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
    svgd_dro_model.fit(train_features_np, train_labels_np)
    
    print("\n--- Evaluating Models ---")
    avg_feature_norm = np.mean(np.linalg.norm(test_features_np, axis=1))
    print(f"Average L2 norm of test set features: {avg_feature_norm:.2f}")
    perturbation_levels = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    epsilon_values = [level * avg_feature_norm for level in perturbation_levels]
    
    all_results = {}
    models_to_evaluate = {
        "SAA": saa_model_wrapper,
        "Dual": sl_dro_model,
        "Transport": svgd_dro_model,
        "RGO": rgo_dro_model
    }

    for name, model_obj in models_to_evaluate.items():
        print(f"\n--- Evaluating {name} ---")
        results = evaluate_model(model_obj, test_features_np, test_labels_np, pgd_attack, epsilon_values[1:])
        all_results[name] = results

    print("\n\n--- FINAL RESULTS SUMMARY ---")
    header = f"{'Model':<20} | " + " | ".join([f"Acc @ eps={p_level:.3f}" for p_level in perturbation_levels])
    print(header)
    print("-" * len(header))
    for name, results in all_results.items():
        row = f"{name:<20} | "
        acc_values = []
        for p_level in perturbation_levels:
            eps_val = p_level * avg_feature_norm
            closest_eps = min(results.keys(), key=lambda k: abs(k-eps_val))
            acc_values.append(f"{results[closest_eps]:>7.2f}%")
        row += " | ".join(acc_values)
        print(row)
    
    print("\nGenerating results plot...")
    plot_results(all_results, perturbation_levels)

