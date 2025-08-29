import numpy as np
import matplotlib.pyplot as plt
import os

# --- NumPy-based helper classes and functions ---

class RGO:
    def __init__(self, input_dim, num_samples, epsilon, lambda_dro, L=2, rgo_inner_steps=20, rgo_vectorized_max_trials=50):
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.lambda_dro = lambda_dro
        self.L =L
        self.rgo_inner_steps = rgo_inner_steps
        self.rgo_vectorized_max_trials = rgo_vectorized_max_trials

    def sample(self, z_i, theta):
        x_pert = z_i.copy()
        lr = 0.1
        for _ in range(int(self.rgo_inner_steps)):
            grad = -2*(x_pert-theta)/self.lambda_dro + 2 * (x_pert - z_i)
            x_pert -= lr * grad
        x_opt_star = x_pert
        var_rgo = self.epsilon * self.lambda_dro / (2*self.lambda_dro-self.L)
        std_rgo = np.sqrt(var_rgo) if var_rgo > 0 else 0.0
        active_mask = np.ones(self.num_samples, dtype=bool)
        samples = np.tile(x_opt_star, (self.num_samples, 1))
        for _ in range(int(self.rgo_vectorized_max_trials)):
            # Only propose for active samples
            current_indices = np.where(active_mask)[0]
            if len(current_indices) == 0:
                break
            proposals = x_opt_star + np.random.randn(len(current_indices), self.input_dim) * std_rgo
            f_opt = -np.sum((x_opt_star - theta) ** 2) / self.lambda_dro / self.epsilon + np.sum((x_opt_star - z_i) ** 2) / self.epsilon
            f_cand = -np.sum((proposals - theta) ** 2, axis=1) / self.lambda_dro / self.epsilon + np.sum((proposals - z_i) ** 2, axis=1) / self.epsilon
            diff_norm = np.sum((proposals - x_opt_star) ** 2, axis=1)
            exponent = np.clip(-f_cand + f_opt + diff_norm / (2 * var_rgo), a_max=0, a_min=None) if var_rgo > 0 else 0.0
            accept_prob = np.exp(exponent)
            accept = np.random.rand(len(current_indices)) < accept_prob
            samples[current_indices[accept]] = proposals[accept]
            active_mask[current_indices[accept]] = False
        return samples

def rbf_kernel_full_numpy(X):
    N, D = X.shape
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    all_dist_sq = dist_sq.flatten()
    if all_dist_sq.size > 1:
        h2 = 0.5 * np.median(all_dist_sq) / np.log(N + 1.0)
    else:
        h2 = 1.0
    h2 = np.maximum(h2, 1e-6)

    K = np.exp(-dist_sq / (2 * h2))
    grad_K_x = -diff / h2 * K[..., np.newaxis]
    return K, grad_K_x

class SVGD_Sampler:
    def __init__(self, input_dim, num_samples, epsilon, lambda_dro, svgd_inner_steps=50, alpha=0.9, stepsize=0.02):
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.lambda_dro = lambda_dro
        self.svgd_inner_steps = svgd_inner_steps
        self.alpha = alpha
        self.stepsize = stepsize

    def sample(self, z_i, theta):
        def grad_log_prob(x):
            return 2 * (x - theta) / self.epsilon / self.lambda_dro - 2 * (x - z_i) /self.epsilon

        var_rgo = 1.0 / (2 * self.epsilon)
        std_rgo = np.sqrt(var_rgo) if var_rgo > 0 else 0.0
        x = z_i + np.random.randn(self.num_samples, self.input_dim) * std_rgo
        hist_grad = np.zeros_like(x)

        for _ in range(self.svgd_inner_steps):
            N = x.shape[0]
            K, grad_K_x = rbf_kernel_full_numpy(x)
            grad_log_p_at_x = grad_log_prob(x)
            svgd_grad = (K @ grad_log_p_at_x + np.sum(grad_K_x, axis=1)) / N
            hist_grad = self.alpha * hist_grad + (1 - self.alpha) * (svgd_grad**2)
            adj_grad = svgd_grad / (1e-6 + np.sqrt(hist_grad))
            x = x + self.stepsize * adj_grad
        return x

def true_grad(theta, z, lam, epsilon):
    return 2*lam/(lam-1) * (theta - np.mean(z, axis=0))

def dual_approx_grad(theta, z, lam, epsilon, num_sample=8):
    N, d = z.shape
    scale = np.sqrt(epsilon / 2.0)
    samples = z[:, np.newaxis, :] + np.random.randn(N, num_sample, d) * scale
    f_values = np.sum(np.square(samples - theta), axis=2)
    max_f = np.max(f_values, axis=1, keepdims=True)
    exp_values = np.exp((f_values-max_f) / lam/epsilon)
    grad = ((exp_values[:,:,np.newaxis]*(theta-samples)*2).mean(axis=1) / (exp_values.mean(axis=1))[:, np.newaxis]).mean(axis=0)
    return grad

def rgo_approx_grad(theta, z, lam, epsilon, num_sample=8):
    rgo = RGO(input_dim=z.shape[1], num_samples=num_sample, epsilon=epsilon, lambda_dro=lam)
    grads = [2 * np.mean(theta-rgo.sample(z[i], theta), axis=0) for i in range(z.shape[0])]
    return np.mean(grads, axis=0)

def svgd_approx_grad(theta, z, lam, epsilon, num_sample=8):
    svgd = SVGD_Sampler(input_dim=z.shape[1], num_samples=num_sample, epsilon=epsilon, lambda_dro=lam)
    grads = [2 * np.mean(theta-svgd.sample(z[i], theta), axis=0) for i in range(z.shape[0])]
    return np.mean(grads, axis=0)

# --- Updated Visualization Function ---
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def plot_angular_error_boxplot(theta_challenging, z_centers, lam, epsilon):
    """
    Generates a box plot of the angular error for each estimator at a challenging theta.
    """
    print("Generating angular error box plot...")
    repeat_times = 100
    
    dual_errors, rgo_errors, svgd_errors = [], [], []
    true_g = true_grad(theta_challenging, z_centers, lam, epsilon)

    for i in range(repeat_times):
        if (i+1) % 20 == 0: print(f"  ...iteration {i+1}/{repeat_times}")
            
        approx_grads = {
            'dual': dual_approx_grad(theta_challenging, z_centers, lam, epsilon),
            'rgo': rgo_approx_grad(theta_challenging, z_centers, lam, epsilon),
            'svgd': svgd_approx_grad(theta_challenging, z_centers, lam, epsilon)
        }
        
        sim_dual = cosine_similarity(approx_grads['dual'], true_g)
        sim_rgo = cosine_similarity(approx_grads['rgo'], true_g)
        sim_svgd = cosine_similarity(approx_grads['svgd'], true_g)
        
        # Clip similarity to avoid arccos domain errors from float inaccuracies
        dual_errors.append(np.degrees(np.arccos(np.clip(sim_dual, -1.0, 1.0))))
        rgo_errors.append(np.degrees(np.arccos(np.clip(sim_rgo, -1.0, 1.0))))
        svgd_errors.append(np.degrees(np.arccos(np.clip(sim_svgd, -1.0, 1.0))))
        
    plt.figure(figsize=(10, 7))
    plt.boxplot([dual_errors, rgo_errors], labels=['Dual', 'RGO'])
    plt.title(f'Distribution of Angular Error at [{theta_challenging[0]}, {theta_challenging[1]}]($\\lambda = {lam}, \\epsilon = {epsilon}$)', fontsize=16)
    plt.ylabel('Angular Error (degrees)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'angular_error/angular_error_[{theta_challenging[0]}, {theta_challenging[1]}]_{lam}_{epsilon}.png')
    print("Generated plot: angular_error_boxplot.png\n")

def plot_gradient_fields_normalized(z_centers, lam, epsilon, num_runs=5):
    """
    Visualizes the normalized (unit vector) true and approximated gradient fields.
    """
    print(f"Generating NORMALIZED gradient fields (averaging over {num_runs} runs)...")
    x = np.linspace(-1, 6, 12)
    y = np.linspace(-1, 6, 12)
    X, Y = np.meshgrid(x, y)
    thetas = np.vstack([X.ravel(), Y.ravel()]).T

    # Calculate true and averaged approximate gradients
    true_grads = np.array([true_grad(t, z_centers, lam, epsilon) for t in thetas])
    avg_grads = {'dual': [], 'rgo': [], 'svgd': []}
    for i, t in enumerate(thetas):
        if (i+1) % 20 == 0: print(f"  ...processing grid point {i+1}/{len(thetas)}")
        dual_runs = [dual_approx_grad(t, z_centers, lam, epsilon) for _ in range(num_runs)]
        rgo_runs = [rgo_approx_grad(t, z_centers, lam, epsilon) for _ in range(num_runs)]
        svgd_runs = [svgd_approx_grad(t, z_centers, lam, epsilon) for _ in range(num_runs)]
        avg_grads['dual'].append(np.mean(dual_runs, axis=0))
        avg_grads['rgo'].append(np.mean(rgo_runs, axis=0))
        avg_grads['svgd'].append(np.mean(svgd_runs, axis=0))

    grad_fields = {
        "True Gradient": true_grads,
        "Dual (Avg)": np.array(avg_grads['dual']),
        "RGO (Avg)": np.array(avg_grads['rgo']),
        "SVGD (Avg)": np.array(avg_grads['svgd'])
    }
    
    # --- NORMALIZATION STEP ---
    normalized_fields = {}
    for title, grads in grad_fields.items():
        norms = np.linalg.norm(grads, axis=1)
        # Add a small epsilon to avoid division by zero for null vectors
        norms[norms == 0] = 1e-8
        normalized_fields[title] = grads / norms[:, np.newaxis]

    # --- Plotting Step ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    fig.suptitle('Normalized Gradient Fields (Direction-Only Comparison)', fontsize=20)
    for i, (title, norm_grads) in enumerate(normalized_fields.items()):
        ax = axes[i]
        ax.quiver(thetas[:, 0], thetas[:, 1], norm_grads[:, 0], norm_grads[:, 1], color='teal')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$\\theta_1$')
        ax.set_ylabel('$\\theta_2$')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('gradient_fields_normalized.png')
    print("\nGenerated plot: gradient_fields_normalized.png")

def optimize_theta(theta, z_centers, lam, epsilon, method, itr=100, lr=0.01, num_sample=20):
    """
    Optimizes theta using different methods.
    """
    if method == 'dual':
        grad_func = dual_approx_grad
    elif method == 'rgo':
        grad_func = rgo_approx_grad
    elif method == 'svgd':
        grad_func = svgd_approx_grad
    elif method == 'true':
        grad_func = true_grad
    else:
        raise ValueError("Unknown method: choose from 'dual', 'rgo', or 'svgd'.")
    his = [np.mean(np.sum((theta - z_centers)**2, axis=1))]
    for _ in range(itr):  # Number of optimization steps
        grad = grad_func(theta, z_centers, lam, epsilon)
        theta -= lr * grad
        his.append(np.mean(np.sum((theta - z_centers)**2, axis=1)))
    return theta, his

if __name__ == '__main__':
    z_main = np.array([[4.0, 5.0],[5.0, 4.0], [4.0, 4.0], [5,5], [-2.0, -2.0]]) 
    lambdas_to_test = [5, 10, 20]
    epsilons_to_test = [0.001, 0.01, 0.1]
    theta_to_test = np.array([[4.0, 4.0], [0.0, 0.0], [-1.0, -1.0]])  # Example theta for testing

    save_dir = "gaussian_plots"
    os.makedirs(save_dir, exist_ok=True)
    # --- Loop over parameters and generate plots ---
    for lam in lambdas_to_test:
        for eps in epsilons_to_test:
            print(f"--- Running simulation for lambda={lam}, epsilon={eps} ---")
            
            # Initial theta for optimization
            theta_initial = np.array([0.0, 0.0])

            # Run optimization for each method
            optimized_theta_dual, dual_history = optimize_theta(theta_initial.copy(), z_main, lam, eps, method='dual', itr=100)
            optimized_theta_rgo, rgo_history = optimize_theta(theta_initial.copy(), z_main, lam, eps, method='rgo', itr=100)
            optimized_theta_svgd, svgd_history = optimize_theta(theta_initial.copy(), z_main, lam, eps, method='svgd', itr=100)
            optimized_theta_true, true_history = optimize_theta(theta_initial.copy(), z_main, lam, eps, method='true', itr=100)
            
            # Plotting the results
            plt.style.use('jz.mplstyle')
            plt.figure() 

            plt.plot(dual_history, label='Dual')
            plt.plot(rgo_history, label='RGO')
            plt.plot(svgd_history, label='Transport')
            plt.plot(true_history, label='True', linestyle='--', color='black')
            
            plt.xlabel('iteration')
            plt.ylabel(r'loss') #(E$[\|z-\theta \|^2_2]$)
            plt.title(f'$\\lambda$={lam}, $\\epsilon$={eps}')
            plt.legend(loc='best', fontsize=9)

            # Save the figure with a descriptive name
            plot_filename = f"convergence_lam_{lam}_eps_{eps}.png"
            plt.tight_layout()
            filepath = os.path.join(save_dir, plot_filename)
            plt.savefig(filepath)
            print(f"Convergence plot saved as '{plot_filename}'.\n")
            plt.close() # Close the figure to free up memory
            for i in range(len(theta_to_test)):
                plot_angular_error_boxplot(theta_to_test[i], z_centers=z_main, lam=lam, epsilon=eps)

