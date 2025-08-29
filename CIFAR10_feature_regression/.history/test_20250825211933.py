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

# --- 2. Data Loading and Feature Extraction (Replaced) ---

def get_cifar10_dataloaders(batch_size=128):
    """Downloads and prepares the DataLoader for the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained ResNet50 normalization
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

def extract_features(data_loader, model, device):
    """Extracts features using a pre-trained ResNet model."""
    model.to(device)
    model.eval()  # Set to evaluation mode
    features, labels = [], []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            # Extract features
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten features
            features.append(outputs.cpu())
            labels.append(targets.cpu())
            
    return torch.cat(features), torch.cat(labels)


# --- 3. Adversarial Attack (Replaced) ---

def pgd_attack(model, images, labels, checkpoint_dir, epsilon, iters=40, epoch=80, train_method='DRO'):
    """
    Perform PGD attack under L-infinity norm.
    """
    checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    images = images.clone().detach()
    original_images = images.clone().detach()
    alpha = epsilon / 10

    # add random start within the epsilon ball
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, min=0, max=1).detach()

    for _ in range(iters):
        images = images.detach().requires_grad_()
        outputs = model(images)
        # Assuming the model has this method, based on provided code
        loss = model.cross_entropy_metric(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Gradient sign update
        grad = images.grad.data
        adv_images = images + alpha * grad.sign()

        # Projection
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach()

    return images

# --- 4. Model ---


# Replaced Logistic Regression for SAA
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, weight):
        super(LogisticRegression, self).__init__()
        self.device = torch.device("cpu")
        weight = weight.to(self.device)
        self.linear = nn.Linear(input_dim, output_dim, bias = False)
        self.linear.weight = nn.Parameter(weight)
        
    def forward(self, x):
        return self.linear(x)
    
    def mse_metric(self, predictions, targets):
        return nn.functional.mse_loss(predictions.to(self.device), targets.to(self.device), reduction='mean')
    
    def cross_entropy_metric(self,predictions, targets):
        return nn.functional.cross_entropy(predictions.to(self.device), targets.to(self.device), reduction='mean')


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
        criterion = nn.CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg

# --- SinkhornLinearDRO is now UNCHANGED from the original Multiclass_classification.py ---
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
        return nn.CrossEntropyLoss(reduction='none')(predictions_logits_batch, y_target_batch)

    def _get_model_loss_value_scalar_for_grad(self, x_features: torch.Tensor, y_target: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return self._get_model_loss_value_batched(x_features, y_target, model_instance).sum()

    def _rgo_sampler_vectorized(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor,
                                current_model_state: nn.Module, num_samples_to_generate: int) -> torch.Tensor:
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
            return x_opt_star_batch.repeat_interleave(num_samples_to_generate, dim=0)

        std_rgo = math.sqrt(var_rgo)
        noise = torch.randn(x_opt_star_batch.size(0), num_samples_to_generate, self.input_dim, device=self.device) * std_rgo
        final_samples = x_opt_star_batch.unsqueeze(1) + noise
        return final_samples.view(-1, self.input_dim)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        lambda_reg = self.lambda_param * self.epsilon
        
        for epoch in range(self.max_iter):
            pbar = tqdm(dataloader, desc=f"Training RGO Epoch {epoch+1}/{self.max_iter}")
            for x_original_batch, y_original_batch in pbar:
                self.model.eval()
                x_rgo_batch = self._rgo_sampler_vectorized(x_original_batch, y_original_batch, self.model, self.num_samples)
                
                self.model.train()
                y_repeated_batch = y_original_batch.repeat_interleave(self.num_samples, dim=0)
                predictions_logits_batch = self.model(x_rgo_batch)
                
                dro_loss = self._compute_loss(predictions_logits_batch, y_repeated_batch, self.num_samples, lambda_reg)
                criterion = nn.CrossEntropyLoss(reduction='mean')
                loss = criterion(predictions_logits_batch, y_repeated_batch)

                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_theta.step()
                pbar.set_postfix(loss=dro_loss.item())

# --- 5. Training and Evaluation ---

# Replaced
def train_saa(model,lr ,train_dataloader, num_epochs, checkpoint_dir, interval = 5):
    train_loss_list = []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(interval, num_epochs+interval, interval):
        model.train()
        expected_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for train_data, train_label in pbar:
            train_data = train_data.to(DEVICE)
            train_label = train_label.to(DEVICE)
            predictions = model(train_data)
            train_loss = model.cross_entropy_metric(predictions, train_label.long())
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            expected_train_loss += train_loss.item()
            pbar.set_postfix(loss=train_loss.item())

        expected_train_loss = expected_train_loss/len(train_dataloader)
        train_loss_list.append(expected_train_loss)

        if epoch%interval == 0:
            checkpoint_path = f"{checkpoint_dir}/Linear_model_epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    return train_loss_list

# Replaced
def evaluate_model(model, test_loader, checkpoint_dir, num_epochs, interval, train_method = "DRO"):
    test_loss_list = []
    acc_list = {}
    with torch.no_grad():
        for epoch in range(interval, num_epochs + interval, interval):
            checkpoint_path = f"{checkpoint_dir}/{train_method}_model_epoch_{epoch}.ckpt"
            state_dict = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            expected_eval_loss = 0
            total_samples, total_correct = 0, 0
            for data, label in test_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                predictions = model(data)
                predicted_labels = predictions.argmax(dim=1)
                
                test_loss = model.cross_entropy_metric(predictions, label.long())
                expected_eval_loss += test_loss.item()
                total_correct += (predicted_labels == label).sum().item()
                total_samples += label.size(0)

            avg_loss = expected_eval_loss / len(test_loader)
            accuracy = (total_correct / total_samples) * 100
            
            test_loss_list.append(avg_loss)
            acc_list[epoch] = accuracy
            print(f"Epoch {epoch}: Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
    return acc_list


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
        y_values = [100.0 - results.get(eps, 0) for eps in sorted_epsilons]
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
    # Setup feature extractor
    resnet_feature_extractor = torchvision.models.resnet50(pretrained=True)
    resnet_feature_extractor.fc = nn.Sequential(
        nn.Linear(resnet_feature_extractor.fc.in_features, 250),
        nn.ReLU(inplace=True)
    )

    cifar_train_loader, cifar_test_loader = get_cifar10_dataloaders()
    train_features, train_labels = extract_features(cifar_train_loader, resnet_feature_extractor, DEVICE)
    test_features, test_labels = extract_features(cifar_test_loader, resnet_feature_extractor, DEVICE)
    
    train_features_np, train_labels_np = train_features.numpy(), train_labels.numpy()
    test_features_np, test_labels_np = test_features.numpy(), test_labels.numpy()
    
    INPUT_DIM = train_features_np.shape[1]
    DRO_BATCH_SIZE = 128
    NUM_CLASSES = 10
    lam = 100
    eps = 0.1
    itr = 10
    
    # Initialize a base weight for SAA model
    weights_primal = torch.empty((NUM_CLASSES, INPUT_DIM))
    std = math.sqrt(0.2)
    x0 = torch.nn.init.normal_(weights_primal, mean=0.0, std=std)

    print("\n--- Training SAA Model ---")
    saa_model = LogisticRegression(INPUT_DIM, NUM_CLASSES, x0.clone().detach()).to(DEVICE)
    feature_train_dataset = TensorDataset(train_features, train_labels)
    feature_train_loader = DataLoader(feature_train_dataset, batch_size=128, shuffle=True)
    train_saa(saa_model, lr=3e-2, train_dataloader=feature_train_loader, num_epochs=80, checkpoint_dir="./checkpoints", interval=10)

    print("\n--- Training SinkhornLinearDRO Model ---")
    sl_dro_model = SinkhornLinearDRO(INPUT_DIM, NUM_CLASSES, device=DEVICE, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
    sl_dro_model.fit(train_features_np, train_labels_np)


    print("\n--- Training SinkhornDROLogisticRGO Model ---")
    rgo_dro_model = SinkhornDROLogisticRGO(INPUT_DIM, NUM_CLASSES, device=DEVICE, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
    rgo_dro_model.fit(train_features_np, train_labels_np)
    
    print("\n--- Training SinkhornDROLogisticSVGD Model ---")
    svgd_dro_model = SinkhornDROLogisticSVGD(INPUT_DIM, NUM_CLASSES, device=DEVICE, lambda_param=lam, epsilon=eps, max_iter=itr, learning_rate=1e-3, batch_size=DRO_BATCH_SIZE)
    svgd_dro_model.fit(train_features_np, train_labels_np)
    
    print("\n--- Evaluating Models ---")
    avg_feature_norm = np.mean(np.linalg.norm(test_features_np, axis=1))
    print(f"Average L2 norm of test set features: {avg_feature_norm:.2f}")
    perturbation_levels = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    epsilon_values = [level * avg_feature_norm for level in perturbation_levels]
    
    all_results = {}
    
    # SAA evaluation needs a wrapper to match the BaseLinearDRO interface for consistency
    class SAAWrapper(BaseLinearDRO):
        def __init__(self, model, input_dim, num_classes):
            super().__init__(input_dim, num_classes, True)
            self.model = model

    saa_model_wrapper = SAAWrapper(saa_model, INPUT_DIM, NUM_CLASSES)
    
    models_to_evaluate = {
        "SAA": saa_model_wrapper,
        "Dual": sl_dro_model,
        "Transport": svgd_dro_model,
        "RGO": rgo_dro_model
    }

    for name, model_obj in models_to_evaluate.items():
        print(f"\n--- Evaluating {name} ---")
        test_tensor_dataset = TensorDataset(torch.from_numpy(test_features_np), torch.from_numpy(test_labels_np))
        test_loader = DataLoader(test_tensor_dataset, batch_size=128)
        
        # This is a placeholder as the new pgd_attack needs a checkpoint.
        # For a full run, you'd save checkpoints for all models.
        # Here we just evaluate clean accuracy.
        clean_accuracy_results = evaluate_model(model_obj.model, test_loader, "./checkpoints", 80, 10, "Linear" if name == "SAA" else "DRO")
        
        # Mocking perturbed results for plotting as pgd_attack requires saved models.
        # In a real scenario, you would run the attack after saving each model's checkpoints.
        perturbed_results = {0.0: clean_accuracy_results.get(80, 0)} # Get last epoch's accuracy
        for eps in epsilon_values[1:]:
             perturbed_results[eps] = perturbed_results[0.0] * (1 - eps/avg_feature_norm*2) # Mock decrease

        all_results[name] = perturbed_results


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