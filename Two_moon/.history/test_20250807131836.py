import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import warnings
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import f1_score
from typing import Tuple, List, Dict

# --- Setup and Base Classes ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Classifier(nn.Module):
    """A non-linear classifier for the DRO methods."""
    def __init__(self, input_dim, output_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class BaseDRO:
    """A base class to provide a common interface for all DRO methods."""
    def __init__(self, input_dim: int, fit_intercept: bool = True):
        self.input_dim = input_dim
        self.fit_intercept = fit_intercept
        self.model = None
        self.device = device

    def _validate_inputs(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Input data and labels must be numpy arrays.")

    def _create_dataloader(self, X, y, batch_size, y_is_long=True):
        X_tensor = self._to_tensor(X) 
        y_tensor = torch.from_numpy(y).to(self.device)
        if y_is_long:
            y_tensor = y_tensor.long()
        else:
            y_tensor = y_tensor.float()
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _to_tensor(self, data):
        return torch.from_numpy(data).float().to(self.device)

    def _extract_parameters(self):
        if self.model is None: return {}
        return {name: param.detach().cpu().numpy() for name, param in self.model.named_parameters()}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[float], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        raise NotImplementedError("The fit method must be implemented by subclasses.")

# Helper classes for particle-based methods
class AdjustedSteinTransport:
    def __init__(self, lam=1e-2, n_steps=50, n_svgd=1, dt=0.02, dt_svgd=0.02):
        self.lam, self.n_steps, self.n_svgd, self.dt, self.dt_svgd = lam, n_steps, n_svgd, dt, dt_svgd
    def _rbf_kernel(self, X):
        N, D = X.shape; pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
        h2 = max(0.5 * np.median(pairwise_sq_dists) / np.log(N + 1), 1e-6)
        K = np.exp(-pairwise_sq_dists / (2 * h2)); X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return K, -X_diff / h2 * K[..., np.newaxis], X_diff / h2 * K[..., np.newaxis], (D/h2 - pairwise_sq_dists/h2**2) * K
    def update(self, X0, grad_log_prior, nll, grad_nll):
        X, N = np.copy(X0), X0.shape[0]
        for n in range(self.n_steps):
            grad_log_pi_t = lambda x: grad_log_prior(x) - (n*self.dt)*grad_nll(x)
            for _ in range(self.n_svgd):
                P_svgd = grad_log_pi_t(X); K_svgd, _, grad_K_y_svgd, _ = self._rbf_kernel(X)
                X += self.dt_svgd * (K_svgd @ P_svgd + np.sum(grad_K_y_svgd, axis=1)) / N
            P = grad_log_pi_t(X); K, gx, gy, trH = self._rbf_kernel(X)
            xi_matrix = np.einsum('id,ijd->ij',P,gy) + np.einsum('jd,ijd->ij',P,gx) + (P@P.T)*K + trH
            h = nll(X).flatten(); h_centered = h - np.mean(h)
            A = xi_matrix / N + self.lam * np.identity(N)
            try: phi = np.linalg.solve(A, h_centered)
            except np.linalg.LinAlgError: phi = np.linalg.pinv(A) @ h_centered
            X += self.dt * ((K * phi[np.newaxis, :]) @ P + np.einsum('j,ijd->id', phi, gy)) / N
        return X

class SVGD:
    def __init__(self): pass
    def _svgd_kernel(self, theta):
        N, _ = theta.shape; pairwise_sq_dists = squareform(pdist(theta, 'sqeuclidean'))
        h2 = max(0.5 * np.median(pairwise_sq_dists) / np.log(N + 1), 1e-6)
        K = np.exp(-pairwise_sq_dists / (2 * h2))
        return K, -(K @ theta - np.sum(K, axis=1, keepdims=True) * theta) / h2
    def update(self, x0, grad_log_prob, n_iter, stepsize, alpha=0.9):
        theta = np.copy(x0); #hist_grad = np.zeros_like(theta)
        for i in range(n_iter):
            k, grad_k = self._svgd_kernel(theta)
            grad_theta = (k @ grad_log_prob(theta) + grad_k) / x0.shape[0]
            # if i == 0: hist_grad = grad_theta ** 2
            # else: hist_grad = alpha * hist_grad + (1-alpha) * (grad_theta**2)
            theta += stepsize * grad_theta #/ (1e-6 + np.sqrt(hist_grad))
        return theta

class ProximalFisherRaoFlow:
    def __init__(self, n_outer_steps=50, dt_outer=0.02, n_inner_steps=4, dt_inner=0.01, clip_value=1e4, alpha=0.9, fudge_factor=1e-6):
        self.n_outer_steps, self.dt_outer = n_outer_steps, dt_outer
        self.n_inner_steps, self.dt_inner = n_inner_steps, dt_inner
        self.clip_value, self.alpha, self.fudge_factor = clip_value, alpha, fudge_factor
    def _rbf_kernel(self, X):
        N, D = X.shape; pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
        h2 = max(0.5 * np.median(pairwise_sq_dists) / np.log(N + 1.0), 1e-8)
        K = np.exp(-pairwise_sq_dists / (2 * h2))
        return K, (X[:, np.newaxis, :] - X[np.newaxis, :, :]) / h2 * K[..., np.newaxis]
    def update(self, X0, grad_log_prior, grad_neg_log_likelihood):
        X = np.copy(X0)
        #historical_grad = np.zeros_like(X)
        for k in range(self.n_outer_steps):
            t_k = self.dt_outer * k
            grad_log_target_k = lambda x: grad_log_prior(x) - t_k * grad_neg_log_likelihood(x)
            for i in range(self.n_inner_steps):
                score_at_X = grad_log_target_k(X)
                np.clip(score_at_X, -self.clip_value, self.clip_value, out=score_at_X)
                K, grad_K_y = self._rbf_kernel(X)
                svgd_grad = (K @ score_at_X + np.sum(grad_K_y, axis=1)) / X.shape[0]
                # if k == 0: historical_grad = svgd_grad ** 2
                # else: historical_grad = self.alpha * historical_grad + (1 - self.alpha) * (svgd_grad ** 2)
                X += self.dt_inner * svgd_grad #/ (self.fudge_factor + np.sqrt(historical_grad))
        return X

class KFR:
    """
    A class to implement the Kernel Fisher-Rao Flow (KFR) algorithm, refactored for consistency.
    Takes and returns (N, D) numpy arrays.
    """
    def __init__(self, kernel_type='imq', nugget=7e-5, device='cpu'):
        self.kernel_type = kernel_type
        self.nugget = nugget
        self.device = device

    def _get_kernel_and_grad(self, x):
        # x is a torch tensor of shape (N, D)
        n_particles, dim = x.shape
        x_col = x.unsqueeze(1)
        x_row = x.unsqueeze(0)
        diffs = x_col - x_row
        sq_dists = torch.sum(diffs**2, dim=2)

        if self.kernel_type == 'rbf':
            h_squared = 0.5 * torch.median(sq_dists.detach()) / torch.log(torch.tensor(n_particles + 1.0)) if n_particles > 1 else 1.0
            h_squared = torch.clamp(h_squared, min=1e-8)
            F_evals = torch.exp(-sq_dists / (2 * h_squared))
            A_tensor = -F_evals.unsqueeze(2) * diffs / h_squared
        elif self.kernel_type == 'imq':
            if n_particles > 1:
                pairwise_dists = torch.sqrt(sq_dists.detach())
                med_dist = torch.median(pairwise_dists)
                h_squared = 0.5 * med_dist**2 / np.log(torch.tensor(n_particles, dtype=torch.float32))
                h_squared = torch.clamp(h_squared, min=1e-8)
            else:
                h_squared = torch.tensor(1.0, device=self.device)
            F_evals = (1 + sq_dists / h_squared)**(-0.5)
            k_cubed = F_evals**3
            A_tensor = -k_cubed.unsqueeze(2) * diffs / h_squared
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")
        return F_evals, A_tensor

    def update(self, X0, log_ratio_func, dt=0.02, n_iter=50):
        # X0 is a numpy array of shape (N, D)
        x = torch.from_numpy(X0).float().to(self.device)
        n_particles, dim = x.shape

        for step in range(n_iter):
            F_evals, A_tensor = self._get_kernel_and_grad(x)
            
            # log_ratio_func takes a numpy array (N,D) and returns a numpy array (N,)
            log_ratios_np = log_ratio_func(x.cpu().numpy())
            log_ratios = torch.from_numpy(log_ratios_np).float().to(self.device)
            
            W = (log_ratios - torch.mean(log_ratios)) * dt / n_particles

            b = F_evals @ W
            M_t = torch.einsum('ijd,ikd->jk', A_tensor, A_tensor) / n_particles
            M_t_reg = M_t + self.nugget * torch.eye(n_particles, device=self.device, dtype=x.dtype)
            s_opt = torch.linalg.solve(M_t_reg, b)
            
            update_step = torch.einsum('ikd,k->id', A_tensor, s_opt)
            x = x + update_step
            
        return x.cpu().numpy()

# Generic class for particle-based primal DRO methods
class PrimalParticleDRO(BaseDRO):
    """
    A base class for primal-based particle DRO methods.
    """
    def __init__(self, input_dim: int, fit_intercept: bool = True, lambda_dro: float = 100.0, 
                 max_iter: int = 100, learning_rate: float = 0.01, batch_size: int = 32):
        super().__init__(input_dim, fit_intercept)
        self.lambda_dro = lambda_dro
        self.max_iter, self.learning_rate = max_iter, learning_rate
        self.model = Classifier(self.input_dim, output_dim=2).to(self.device)
        self.batch_size = batch_size

    def _get_sampler(self, x_batch, y_batch, model, epoch):
        raise NotImplementedError
    
    def fit(self, X, y):
        """
        Trains the model using batched data to find a distributionally robust solution.
        """
        self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y.flatten(), batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_history = []
        last_epoch_samples = []

        for epoch in tqdm(range(self.max_iter), desc=f"Training {self.__class__.__name__}", leave=False):
            epoch_losses = []
            for x_batch, y_batch in dataloader:
                self.model.eval()
                sampler = self._get_sampler(x_batch, y_batch, self.model, epoch)
                worst_case_samples_tensor = sampler()
                
                # Capture samples and their labels from the last epoch
                if epoch == self.max_iter - 1:
                    original_points = x_batch.cpu().numpy()
                    perturbed_points = worst_case_samples_tensor.cpu().detach().numpy()
                    labels = y_batch.cpu().numpy()
                    last_epoch_samples.append((original_points, perturbed_points, labels))

                self.model.train()
                
                preds = self.model(worst_case_samples_tensor)
                # Repeat y_batch to match the shape of predictions from perturbed samples
                y_batch_repeated = y_batch.repeat(worst_case_samples_tensor.shape[0] // x_batch.shape[0])
                loss = nn.CrossEntropyLoss()(preds, y_batch_repeated)
                
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_losses.append(loss.item())
            
            loss_history.append(np.mean(epoch_losses))
        return self._extract_parameters(), loss_history, last_epoch_samples

    def _get_kde_log_prior_grad_fn(self, X_kde):
        """
        Creates a function that computes the gradient of the log-KDE prior.
        """
        N_kde, _ = X_kde.shape
        pairwise_sq_dists = squareform(pdist(X_kde, 'sqeuclidean'))
        med_sq = np.median(pairwise_sq_dists)
        sigma2 = med_sq / (2 * np.log(N_kde)) if N_kde > 1 else 1e-6
        sigma2 = max(sigma2, 1e-6)

        def grad_log_prior(X):
            diff = X[:, np.newaxis, :] - X_kde[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            weights = np.exp(-sq_dists / (2 * sigma2))
            sum_weights = np.sum(weights, axis=1, keepdims=True)
            sum_weights[sum_weights < 1e-9] = 1e-9
            norm_weights = weights / sum_weights
            weighted_avg_X_kde = np.sum(norm_weights[..., np.newaxis] * X_kde[np.newaxis, :, :], axis=1)
            grad = (weighted_avg_X_kde - X) / sigma2
            return grad
        return grad_log_prior

# --- Specific DRO Classifier Implementations ---

class KLDRO_AST(PrimalParticleDRO):
    """
    DRO Classifier using Adjusted Stein Transport (AST) with a KDE-based prior.
    """
    def __init__(self, input_dim: int, fit_intercept: bool = True, ast_n_steps: int = 50, **kwargs):
        super().__init__(input_dim, fit_intercept, **kwargs)
        self.ast_n_steps = ast_n_steps

    def _get_sampler(self, x_batch, y_batch, model, epoch):
        n_steps = int(self.ast_n_steps * epoch / max(1, self.max_iter-1)) if self.max_iter > 1 else self.ast_n_steps
        ast_params = {'lam': 0.1, 'n_steps': n_steps, 'n_svgd': 1, 'dt': 1/self.ast_n_steps, 'dt_svgd': 1/self.ast_n_steps}
        ast = AdjustedSteinTransport(**ast_params) 
        grad_log_prior = self._get_kde_log_prior_grad_fn(x_batch.cpu().numpy())
        
        def nll(x_np):
            x = self._to_tensor(x_np)
            # The particles x_np have the same batch dimension as y_batch
            return (-F.cross_entropy(model(x), y_batch, reduction='none') / self.lambda_dro).detach().cpu().numpy()

        def grad_nll(x_np):
            x = self._to_tensor(x_np).requires_grad_(True)
            # The particles x_np have the same batch dimension as y_batch
            loss = -F.cross_entropy(model(x), y_batch, reduction='sum') / self.lambda_dro
            return torch.autograd.grad(loss, x)[0].detach().cpu().numpy()
        
        X0 = x_batch.cpu().numpy()
        return lambda: self._to_tensor(ast.update(X0, grad_log_prior, nll, grad_nll))

class KLDRO_SVGD(PrimalParticleDRO):
    """
    DRO Classifier using Stein Variational Gradient Descent (SVGD) with a KDE-based prior.
    """
    def __init__(self, input_dim: int, fit_intercept: bool = True, svgd_n_iter: int = 200, **kwargs):
        super().__init__(input_dim, fit_intercept, **kwargs)
        self.svgd_n_iter = svgd_n_iter

    def _get_sampler(self, x_batch, y_batch, model, epoch):
        n_iter = int(self.svgd_n_iter * epoch / max(1, self.max_iter-1)) if self.max_iter > 1 else self.svgd_n_iter
        svgd = SVGD()
        grad_log_prior = self._get_kde_log_prior_grad_fn(x_batch.cpu().numpy())

        def grad_log_prob(x_np):
            grad_prior = grad_log_prior(x_np)
            x = self._to_tensor(x_np).requires_grad_(True)
            # The particles x_np have the same batch dimension as y_batch
            loss = -F.cross_entropy(model(x), y_batch, reduction='sum') / self.lambda_dro
            grad_likelihood = torch.autograd.grad(loss, x)[0].detach().cpu().numpy()
            return grad_prior - grad_likelihood

        X0 = x_batch.cpu().numpy()
        return lambda: self._to_tensor(svgd.update(X0, grad_log_prob, n_iter=n_iter, stepsize=1e-2))

class KLDRO_PFR(PrimalParticleDRO):
    """
    DRO Classifier using Proximal Fisher-Rao Flow (PFR) with a KDE-based prior.
    """
    def __init__(self, input_dim: int, fit_intercept: bool = True, pfr_n_steps: int = 50, pfr_inner_steps: int = 4, **kwargs):
        super().__init__(input_dim, fit_intercept, **kwargs)
        self.pfr_n_steps = pfr_n_steps
        self.pfr_inner_steps = pfr_inner_steps

    def _get_sampler(self, x_batch, y_batch, model, epoch):
        n_outer_steps = int(self.pfr_n_steps * epoch/max(1, self.max_iter - 1)) if self.max_iter > 1 else self.pfr_n_steps
        pfr_params = {
            'n_outer_steps': n_outer_steps,
            'dt_outer': 1.0 / self.pfr_n_steps if self.pfr_n_steps > 0 else 0.02,
            'n_inner_steps': self.pfr_inner_steps
        }
        pfr = ProximalFisherRaoFlow(**pfr_params)
        
        grad_log_prior = self._get_kde_log_prior_grad_fn(x_batch.cpu().numpy())

        def grad_neg_log_likelihood(x_np):
            x = self._to_tensor(x_np).requires_grad_(True)
            # The particles x_np have the same batch dimension as y_batch
            loss = -F.cross_entropy(model(x), y_batch, reduction='sum') / self.lambda_dro
            return torch.autograd.grad(loss, x)[0].detach().cpu().numpy()

        X0 = x_batch.cpu().numpy()
        return lambda: self._to_tensor(pfr.update(X0, grad_log_prior, grad_neg_log_likelihood))

class KLDRO_KFR(PrimalParticleDRO):
    """
    DRO Classifier using Kernel Fisher-Rao Flow (KFR) with a KDE-based prior.
    """
    def __init__(self, input_dim: int, fit_intercept: bool = True, kfr_n_iter: int = 50, **kwargs):
        super().__init__(input_dim, fit_intercept, **kwargs)
        self.kfr_n_iter = kfr_n_iter

    def _get_sampler(self, x_batch, y_batch, model, epoch):
        n_iter = int(self.kfr_n_iter * epoch / max(1, self.max_iter-1)) if self.max_iter > 1 else self.kfr_n_iter
        kfr = KFR(device=self.device)

        def log_ratio_func(x_np):
            x = self._to_tensor(x_np)
            with torch.no_grad():
                log_ratios = -F.cross_entropy(model(x), y_batch, reduction='none') / self.lambda_dro
            return log_ratios.cpu().numpy()

        X0 = x_batch.cpu().numpy()
        return lambda: self._to_tensor(kfr.update(X0, log_ratio_func, dt=1/self.kfr_n_iter, n_iter=n_iter))

class KLDRO(BaseDRO):
    """
    DRO Classifier using the dual formulation for KL-divergence.
    Objective: min_theta {lambda * log(E_P_0 [exp(f_theta(x)/lambda)])}
    """
    def __init__(self, input_dim: int, fit_intercept: bool = True, lambda_dro: float = 1.0,
                 max_iter: int = 100, learning_rate: float = 0.01):
        super().__init__(input_dim, fit_intercept)
        self.lambda_dro = lambda_dro
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.model = Classifier(self.input_dim, output_dim=2).to(self.device)

    def fit(self, X, y, batch_size=32):
        self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y.flatten(), batch_size=batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_history = []

        for epoch in tqdm(range(self.max_iter), desc=f"Training {self.__class__.__name__}", leave=False):
            epoch_losses = []
            for x_batch, y_batch in dataloader:
                self.model.train()
                
                preds = self.model(x_batch)
                per_sample_losses = F.cross_entropy(preds, y_batch, reduction='none')
                
                v = per_sample_losses / self.lambda_dro
                m = torch.max(v)
                log_sum_exp = m + torch.log(torch.mean(torch.exp(v - m)))
                objective = self.lambda_dro * log_sum_exp

                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
                
                epoch_losses.append(objective.item())
            
            loss_history.append(np.mean(epoch_losses))
        # This method doesn't generate explicit perturbed samples, so return an empty list
        return self._extract_parameters(), loss_history, []

# --- Main Comparison Script ---

def create_dataset(n_samples=500, imbalance_ratio=0.8, random_state=42):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples*2, noise=0.15, random_state=random_state)
    X_pos, y_pos, X_neg, y_neg = X[y == 1], y[y == 1], X[y == 0], y[y == 0]
    n_pos, n_neg = int(n_samples * imbalance_ratio), int(n_samples * (1-imbalance_ratio))
    X_imbalanced = np.vstack([X_pos[:n_pos], X_neg[:n_neg]])
    y_imbalanced = np.hstack([y_pos[:n_pos], y_neg[:n_neg]])
    shuffle_idx = np.random.permutation(len(X_imbalanced))
    return X_imbalanced[shuffle_idx], y_imbalanced[shuffle_idx]

def get_true_boundary_model(input_dim):
    """Trains a standard classifier on noise-free data to get an ideal boundary."""
    print("--- Training True Boundary Model ---")
    from sklearn.datasets import make_moons
    X_true, y_true = make_moons(n_samples=2000, noise=0.0, random_state=42)
    true_model = Classifier(input_dim).to(device)
    optimizer = optim.Adam(true_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_true).float().to(device), 
                                      torch.from_numpy(y_true).long().to(device)), 
                        batch_size=128, shuffle=True)
    
    true_model.train()
    for _ in range(50): # Train for 50 epochs
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(true_model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    true_model.eval()
    return true_model

def visualize_all_boundaries(models: Dict[str, nn.Module], true_boundary_model, X, y, title, save_path="."):
    """Creates a single plot with contour lines for each model's boundary."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(X[y==1, 0], X[y==1, 1], c='darkorange', marker='o', edgecolors='k', label='Positive Data', alpha=0.5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Negative Data', alpha=0.5)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    with torch.no_grad():
        Z_true = true_boundary_model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    ax.contour(xx, yy, Z_true, levels=[0.5], colors=['black'], linestyles=['-'], linewidths=3)

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    linestyles = ['--', ':', '-.', '--', ':', '-.']
    
    for i, (name, model) in enumerate(models.items()):
        if model is None: continue
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[colors[i]], linestyles=linestyles[i % len(linestyles)], linewidths=2.5)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='black', lw=3, linestyle='-', label='True Boundary')]
    legend_elements.extend([Line2D([0], [0], color=colors[i], lw=2.5, linestyle=linestyles[i % len(linestyles)], label=name) 
                           for i, name in enumerate(models.keys())])
    ax.legend(handles=legend_elements, loc='best')
    ax.set_title(title)
    plt.savefig(f"{save_path}/KL_boundary_{title.replace(' ', '_')}.png")
    plt.close(fig)

def visualize_perturbed_samples(perturbed_data: Dict[str, list], X_train: np.ndarray, y_train: np.ndarray, title: str, save_path="."):
    """
    Visualizes the perturbed samples from the last epoch for each method,
    coloring them by their class label.
    
    Args:
        perturbed_data: A dictionary where keys are method names and values are lists of
                        (original_points, perturbed_points, labels) tuples from the last epoch.
        X_train: The original training data.
        y_train: The original training labels.
        title: The title for the plot.
        save_path: The directory to save the plot.
    """
    # Filter out methods with no samples (like the dual KLDRO)
    methods_with_samples = {k: v for k, v in perturbed_data.items() if v}
    num_methods_with_samples = len(methods_with_samples)
    
    if num_methods_with_samples == 0:
        print("No methods with perturbed samples to visualize.")
        return

    # Determine grid size for subplots
    cols = int(np.ceil(np.sqrt(num_methods_with_samples)))
    rows = int(np.ceil(num_methods_with_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    axes = axes.flatten()

    for i, (name, samples) in enumerate(methods_with_samples.items()):
        ax = axes[i]
        
        # Plot original data points
        ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='darkorange', marker='o', edgecolors='k', label='Original Positive', alpha=0.3)
        ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='dodgerblue', marker='o', edgecolors='k', label='Original Negative', alpha=0.3)
        
        # Aggregate all samples for this method
        all_originals = np.concatenate([s[0] for s in samples])
        all_perturbed = np.concatenate([s[1] for s in samples])
        all_labels = np.concatenate([s[2] for s in samples])

        # Create masks for positive and negative classes
        pos_mask = (all_labels == 1)
        neg_mask = (all_labels == 0)

        # Plot perturbed points with different colors based on their class
        ax.scatter(all_perturbed[pos_mask, 0], all_perturbed[pos_mask, 1], c='orange', s=15, alpha=0.7, label='Perturbed Positive')
        ax.scatter(all_perturbed[neg_mask, 0], all_perturbed[neg_mask, 1], c='blue', s=15, alpha=0.7, label='Perturbed Negative')
        
        # Draw lines from original to perturbed points
        # Repeat each original point to match the number of perturbed points per original point
        num_perturbed_per_original = len(all_perturbed) // len(all_originals)
        all_originals_repeated = np.repeat(all_originals, num_perturbed_per_original, axis=0)

        for j in range(len(all_perturbed)):
             ax.plot([all_originals_repeated[j, 0], all_perturbed[j, 0]], 
                     [all_originals_repeated[j, 1], all_perturbed[j, 1]], 
                     'k-', lw=0.5, alpha=0.3)

        ax.set_title(name)
        ax.legend()

    # Hide unused subplots (if any)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_path}/KL_perturbed_samples_{title.replace(' ', '_')}.png", dpi=300)
    plt.show()
    plt.close(fig)

def plot_final_comparison(scores: Dict[str, List[float]]):
    """Generates a boxplot and prints summary statistics for F1 scores."""
    print(f"\n{'='*25} Final Results over {len(list(scores.values())[0])} Experiments {'='*25}")
    for name, f1_scores in scores.items():
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        print(f"{name:>25} | Average F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(list(scores.values()), labels=list(scores.keys()), patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    ax.set_title('KL Model Performance Comparison (F1-Score)', fontsize=16)
    ax.set_ylabel('Weighted F1-Score', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('final_performance_comparison.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    NUM_EXPERIMENTS = 5 # Set to 1 to generate one set of plots
    lambda_dro_val = 100.0
    input_dim = 2

    all_f1_scores = {
        "KLDRO": [],
        "AST-KLDRO": [],
        "SVGD-KLDRO": [],
        "PFR-KLDRO": [],
        "KFR-KLDRO": [],
    }

    true_boundary_model = get_true_boundary_model(input_dim)

    for i in range(NUM_EXPERIMENTS):
        print(f"\n{'#'*30} Starting Experiment Run {i+1}/{NUM_EXPERIMENTS} {'#'*30}")
        
        X_train, y_train = create_dataset(n_samples=200, imbalance_ratio=0.9, random_state=42 + i)
        X_test, y_test = create_dataset(n_samples=4000, imbalance_ratio=0.5, random_state=101)
        
        # Use the full training set as a single batch for particle methods
        particle_batch_size = len(X_train)
        
        models_to_run = {
            "KFR-KLDRO": KLDRO_KFR(input_dim=input_dim, max_iter=100, lambda_dro=lambda_dro_val, batch_size=particle_batch_size),
            "KLDRO": KLDRO(input_dim=input_dim, max_iter=100, lambda_dro=lambda_dro_val),
            "AST-KLDRO": KLDRO_AST(input_dim=input_dim, max_iter=100, lambda_dro=lambda_dro_val, batch_size=particle_batch_size),
            "SVGD-KLDRO": KLDRO_SVGD(input_dim=input_dim, max_iter=100, lambda_dro=lambda_dro_val, batch_size=particle_batch_size),
            "PFR-KLDRO": KLDRO_PFR(input_dim=input_dim, max_iter=100, lambda_dro=lambda_dro_val, batch_size=particle_batch_size),

        }
        
        trained_models_this_run = {}
        perturbed_samples_this_run = {}

        for name, model_instance in models_to_run.items():
            print(f"\n--- Running: {name} (Run {i+1}) ---")
            try:
                params, loss_history, perturbed_samples = model_instance.fit(X_train, y_train)
                trained_models_this_run[name] = model_instance.model
                perturbed_samples_this_run[name] = perturbed_samples
                
                model_instance.model.eval()
                with torch.no_grad():
                    y_pred = model_instance.model(torch.from_numpy(X_test).float().to(device)).argmax(dim=1).cpu().numpy()
                score = f1_score(y_test, y_pred, average='weighted')
                all_f1_scores[name].append(score)
                print(f"--- Finished: {name} | F1-Score: {score:.4f} ---")
                
            except Exception as e:
                print(f"!!! ERROR running {name}: {e} !!!"); raise e
        
        visualize_all_boundaries(trained_models_this_run, true_boundary_model, X_train, y_train, f"Combined Boundary - Run {i+1}")
        visualize_perturbed_samples(perturbed_samples_this_run, X_train, y_train, f"KL Perturbed Samples - Run {i+1}")


    if NUM_EXPERIMENTS > 1:
        plot_final_comparison(all_f1_scores)
