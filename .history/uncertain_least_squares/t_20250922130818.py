import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp

# --- Experiment Setup & Parameter Definitions ---
# Define the problem's dimensions
DIM_M = 10  # Number of rows for matrix A
DIM_N = 10  # Number of columns for matrix A (dimension of theta)

# Define the number of training/test samples
N_TRAIN_SAMPLES = 20
N_TEST_SAMPLES = 1000

# Define the perturbation range for distribution shift
DELTA_MIN = 0.0
DELTA_MAX = 10.0
DELTA_STEPS = 50

# --- DRO Algorithm Hyperparameters ---
N_EPOCHS = 50 # Number of iterations for the outer optimization
LEARNING_RATE = 0.01 # Learning rate for the outer optimization of theta
LAMBDA_VAL = 0.1 # Regularization parameter lambda
EPSILON = 0.5 # Entropy regularization parameter epsilon
GRAD_CLIP_THRESHOLD = 100.0 # Gradient clipping threshold to prevent explosion

# RGO (Algorithm 2) Specific Parameters
RGO_SMOOTHNESS_L = 0.5 # Assumed smoothness L of the loss function (must satisfy LAMBDA_VAL > L/2)

# WGF (Algorithm 3), WFR (Algorithm 4), WRM Specific Parameters
K_INNER_STEPS = 3000  # Number of inner loop iterations K
INNER_STEP_SIZE = 0.0001 # Inner loop step size eta
WFR_WEIGHT_STEP_SIZE = 0.0016 # Step size for weight updates in WFR

# WGF, WFR, RGO Specific Parameters
M_PARTICLES = 10 # Number of particles (samples) used

# Dual Method Specific Parameters
SINKHORN_SAMPLE_LEVEL = 4 # Monte Carlo sampling level for the Dual method

# --- Generate a Fixed Problem Instance ---
np.random.seed(219)
A0 = np.random.randn(DIM_M, DIM_N)
A1 = np.random.randn(DIM_M, DIM_N)
b = np.random.randn(DIM_M)

# --- Data Generation Functions ---
def generate_training_data(n_samples):
    """Generates training data."""
    return np.random.uniform(-0.5, 0.5, n_samples)

def generate_test_data(n_samples, delta):
    """Generates test data with a distributional shift."""
    lower_bound = -0.5 * (1 + delta)
    upper_bound = 0.5 * (1 + delta)
    return np.random.uniform(lower_bound, upper_bound, n_samples)

# --- Core Function Definitions (Vectorized) ---
def loss_function(theta, z):
    """ Computes the loss f_theta(z) for a single sample or a batch of samples z. """
    z = np.atleast_1d(z)
    # A_z shape: (num_z, DIM_M, DIM_N)
    A_z = A0[np.newaxis, :, :] + z[:, np.newaxis, np.newaxis] * A1[np.newaxis, :, :]
    # residual shape: (num_z, DIM_M)
    residual = A_z @ theta - b[np.newaxis, :]
    # loss shape: (num_z,)
    loss = np.sum(residual**2, axis=1) / DIM_M
    return loss.squeeze()

def loss_grad_theta(theta, z):
    """ Computes the gradient of the loss function with respect to theta, nabla_theta f_theta(z), for one or a batch of z. """
    z = np.atleast_1d(z)
    # A_z shape: (num_z, DIM_M, DIM_N)
    A_z = A0[np.newaxis, :, :] + z[:, np.newaxis, np.newaxis] * A1[np.newaxis, :, :]
    # residual shape: (num_z, DIM_M)
    residual = A_z @ theta - b[np.newaxis, :]
    # grad shape: (num_z, DIM_N)
    grad = 2 * (A_z.transpose(0, 2, 1) @ residual[:, :, np.newaxis]).squeeze(axis=2)
    return grad.squeeze()

def loss_grad_z(theta, z):
    """ Computes the gradient of the loss function with respect to z, nabla_z f_theta(z), for one or a batch of z. """
    z = np.atleast_1d(z)
    # A_z shape: (num_z, DIM_M, DIM_N)
    A_z = A0[np.newaxis, :, :] + z[:, np.newaxis, np.newaxis] * A1[np.newaxis, :, :]
    # residual shape: (num_z, DIM_M)
    residual = A_z @ theta - b[np.newaxis, :]
    # grad_A_z is A1 @ theta, shape: (DIM_M,)
    grad_A_z = A1 @ theta
    # grad shape: (num_z,)
    grad = 2 * np.sum(residual * grad_A_z[np.newaxis, :], axis=1)
    return grad.squeeze()

# --- ERM (Empirical Risk Minimization) Implementation ---
def erm_objective_function(theta, xi_samples):
    """ Calculates the ERM objective function value (average loss). """
    return np.mean(loss_function(theta, xi_samples))

def solve_erm(xi_train_samples):
    """ Solves the ERM problem using an optimizer. """
    initial_theta = np.zeros(DIM_N)
    result = minimize(
        erm_objective_function, initial_theta, args=(xi_train_samples,), method='BFGS')
    if not result.success:
        print("Warning: ERM optimization may not have converged.")
    return result.x

# --- MMD-DRO (Maximum Mean Discrepancy DRO) Implementation ---
def imq_kernel_and_grad_numpy(x, y, c=1.0, beta=-0.5):
    """
    Computes the Inverse Multi-Quadric (IMQ) kernel and its gradient w.r.t. x.
    k(x, y) = (c + ||x - y||^2)^beta
    """
    diffs = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    sq_dists = np.sum(diffs**2, axis=2)
    kernel_matrix = (c + sq_dists)**beta
    grad_multiplier = 2 * beta * (c + sq_dists)**(beta - 1)
    grad_k_x = diffs * grad_multiplier[:, :, np.newaxis]
    return kernel_matrix, grad_k_x

def solve_mmd_dro(xi_train_samples):
    """
    Solves Sinkhorn DRO using an MMD gradient flow sampler.
    This version uses the Adam optimizer for the inner loop to find the
    worst-case particle distribution, providing a more robust and stable optimization.
    """
    theta = np.zeros(DIM_N)
    num_train = len(xi_train_samples)

    # --- Adam Optimizer Hyperparameters for the Inner Loop ---
    # Adam can often handle a larger and more stable step size than simple gradient ascent.
    ADAM_INNER_STEP_SIZE = 0.001
    BETA_1 = 0.9
    BETA_2 = 0.999
    ADAM_EPSILON = 1e-8

    for epoch in range(N_EPOCHS):
        # Initialize particles at the start of each epoch
        particles = np.tile(xi_train_samples[:, np.newaxis], (1, M_PARTICLES))

        # --- Initialize Adam State Variables ---
        # These are reset for each epoch's inner loop optimization.
        # Their shape must match the variable being updated (particles_flat).
        m_adam = np.zeros(num_train * M_PARTICLES)
        v_adam = np.zeros(num_train * M_PARTICLES)

        current_inner_steps = int(max(5, K_INNER_STEPS * (epoch + 1) / N_EPOCHS))

        # The inner loop now runs for a time 't' from 1 to K
        for t in range(1, current_inner_steps + 1):
            particles_flat = particles.flatten()
            
            # Calculate the gradients that form the "forces" on the particles
            loss_grads = loss_grad_z(theta, particles_flat)
            
            particles_reshaped = particles_flat[:, np.newaxis]
            xi_train_reshaped = xi_train_samples[:, np.newaxis]

            _, grad_K_clone = imq_kernel_and_grad_numpy(particles_reshaped, particles_reshaped)
            mmd_term1 = np.mean(grad_K_clone, axis=1).squeeze()

            _, grad_K_orig = imq_kernel_and_grad_numpy(particles_reshaped, xi_train_reshaped)
            mmd_term2 = np.mean(grad_K_orig, axis=1).squeeze()

            # This is the "velocity" from the original implementation.
            # It's the gradient of the objective we want to MAXIMIZE.
            velocity = loss_grads - LAMBDA_VAL * (mmd_term1 - mmd_term2)

            # --- Adam Optimizer Update Logic ---
            # Adam is a minimizer by convention, so we feed it the NEGATIVE gradient.
            g = -velocity

            # 1. Update biased first and second moment estimates
            m_adam = BETA_1 * m_adam + (1 - BETA_1) * g
            v_adam = BETA_2 * v_adam + (1 - BETA_2) * (g**2)

            # 2. Compute bias-corrected moment estimates
            m_hat = m_adam / (1 - BETA_1**t)
            v_hat = v_adam / (1 - BETA_2**t)

            # 3. Compute the final update for the particles
            # The update is subtracted because Adam minimizes.
            update = ADAM_INNER_STEP_SIZE * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
            particles_flat -= update
            
            # Note: The original heuristic noise term is removed, as Adam's adaptive
            # nature provides the necessary optimization stability.

            particles = particles_flat.reshape(num_train, M_PARTICLES)

        # --- Outer Loop Update for Theta (remains unchanged) ---
        all_grads = loss_grad_theta(theta, particles.flatten())
        avg_particle_grads = all_grads.reshape(num_train, M_PARTICLES, -1).mean(axis=1)
        total_grad = avg_particle_grads.sum(axis=0)

        avg_grad = total_grad / num_train
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad

    return theta

# --- RGO (Algorithm 2) Implementation (Vectorized) ---
def solve_rgo(xi_train_samples):
    """ Solves Sinkhorn DRO using the RGO algorithm. """
    theta = np.zeros(DIM_N)
    num_train = len(xi_train_samples)
    
    def rgo_inner_objective(z, xi, current_theta):
        l = loss_function(current_theta, z.flatten()).reshape(z.shape)
        penalty = (z - xi)**2
        return -l / (LAMBDA_VAL * EPSILON) + penalty / EPSILON

    def rgo_inner_objective_grad_z(z, xi, current_theta):
        grad_l_z = loss_grad_z(current_theta, z)
        grad_penalty = 2 * (z - xi)
        return -grad_l_z / (LAMBDA_VAL * EPSILON) + grad_penalty
    
    for epoch in range(N_EPOCHS):
        # Stage 1: Use gradient descent to find z* for all xi simultaneously
        z_k = xi_train_samples.copy()
        for _ in range(K_INNER_STEPS):
            grad = rgo_inner_objective_grad_z(z_k, xi_train_samples, theta)
            z_k -= INNER_STEP_SIZE * grad
        z_star_batch = z_k

        # Stage 2: Vectorized rejection sampling M_PARTICLES times for each xi
        variance = (LAMBDA_VAL * EPSILON) / (2 * LAMBDA_VAL - RGO_SMOOTHNESS_L)
        if variance <= 0: variance = 1e-6
        std_dev = np.sqrt(variance)
        
        z_star_expanded = z_star_batch[:, np.newaxis]
        xi_expanded = xi_train_samples[:, np.newaxis]
        
        final_accepted_samples = np.zeros((num_train, M_PARTICLES))
        active_flags = np.ones((num_train, M_PARTICLES), dtype=bool)
        
        for _ in range(100): # Max 100 attempts for vectorization
            if not np.any(active_flags): break
            
            proposals = np.random.normal(0, std_dev, size=(num_train, M_PARTICLES))
            z_candidates = z_star_expanded + proposals
            
            f_val_candidates = rgo_inner_objective(z_candidates, xi_expanded, theta)
            f_val_star = rgo_inner_objective(z_star_expanded, xi_expanded, theta)
            
            log_accept_prob = -f_val_candidates + f_val_star - (proposals**2 / (2*variance))
            
            acceptance_mask = np.log(np.random.rand(num_train, M_PARTICLES)) < log_accept_prob
            newly_accepted = acceptance_mask & active_flags
            
            final_accepted_samples[newly_accepted] = z_candidates[newly_accepted]
            active_flags[newly_accepted] = False

        if np.any(active_flags):
            final_accepted_samples[active_flags] = z_star_expanded[active_flags]

        all_grads = loss_grad_theta(theta, final_accepted_samples.flatten())
        avg_particle_grads = all_grads.reshape(num_train, M_PARTICLES, -1).mean(axis=1)
        total_grad = avg_particle_grads.sum(axis=0)
        
        avg_grad = total_grad / num_train
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- Wasserstein Gradient Flow (WGF) Implementation (Vectorized) ---
def solve_wgf(xi_train_samples):
    """ Solves Sinkhorn DRO using Wasserstein Gradient Flow (WGF). """
    theta = np.zeros(DIM_N)
    num_train = len(xi_train_samples)
    
    def f_bar_grad_z(z, xi, current_theta):
        grad_f = loss_grad_z(current_theta, z)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    for epoch in range(N_EPOCHS):
        particles = np.tile(xi_train_samples[:, np.newaxis], (1, M_PARTICLES))
        
        for _ in range(K_INNER_STEPS):
            particles_flat = particles.flatten()
            xi_expanded = np.repeat(xi_train_samples, M_PARTICLES)
            
            grad = f_bar_grad_z(particles_flat, xi_expanded, theta)
            noise = np.random.normal(0, 1, size=particles_flat.shape)
            
            updated_particles_flat = particles_flat + INNER_STEP_SIZE * grad + np.sqrt(2 * INNER_STEP_SIZE * LAMBDA_VAL * EPSILON) * noise
            particles = updated_particles_flat.reshape(num_train, M_PARTICLES)
            particles = np.clip(particles, -1, 1)
            
        all_grads = loss_grad_theta(theta, particles.flatten())
        avg_particle_grads = all_grads.reshape(num_train, M_PARTICLES, -1).mean(axis=1)
        total_grad = avg_particle_grads.sum(axis=0)
            
        avg_grad = total_grad / num_train
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- WFR Flow (Algorithm 4) Implementation (Vectorized) ---
def solve_wfr(xi_train_samples):
    """ Solves Sinkhorn DRO using WFR Flow. """
    theta = np.zeros(DIM_N)
    num_train = len(xi_train_samples)
    
    def f_bar_grad_z(z, xi, current_theta):
        grad_f = loss_grad_z(current_theta, z)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    for epoch in range(N_EPOCHS):
        particles = np.tile(xi_train_samples[:, np.newaxis], (1, M_PARTICLES))
        weights = np.full((num_train, M_PARTICLES), 1.0 / M_PARTICLES)
        
        for _ in range(K_INNER_STEPS):
            particles_flat = particles.flatten()
            xi_expanded = np.repeat(xi_train_samples, M_PARTICLES)
            
            grad = f_bar_grad_z(particles_flat, xi_expanded, theta)
            noise = np.random.normal(0, 1, size=particles_flat.shape)
            updated_particles_flat = particles_flat + INNER_STEP_SIZE * grad + np.sqrt(2 * INNER_STEP_SIZE * LAMBDA_VAL * EPSILON) * noise
            particles = updated_particles_flat.reshape(num_train, M_PARTICLES)
            particles = np.clip(particles, -1, 1)
            
            f_bar_val = loss_function(theta, particles.flatten()).reshape(particles.shape) - LAMBDA_VAL * (particles - xi_train_samples[:, np.newaxis])**2
            weights = (weights**(1 - LAMBDA_VAL * EPSILON * WFR_WEIGHT_STEP_SIZE)) * np.exp(WFR_WEIGHT_STEP_SIZE * f_bar_val)
            
            sum_weights = np.sum(weights, axis=1, keepdims=True)
            weights /= (sum_weights + 1e-9)
            
        all_grads = loss_grad_theta(theta, particles.flatten()).reshape(num_train, M_PARTICLES, -1)
        weighted_grads = np.sum(weights[:, :, np.newaxis] * all_grads, axis=1)
        total_grad = np.sum(weighted_grads, axis=0)
        
        avg_grad = total_grad / num_train
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- WRM (Wasserstein Robust Method) Implementation (Vectorized) ---
def solve_wrm(xi_train_samples):
    """ Solves using WRM (deterministic inner optimization). """
    theta = np.zeros(DIM_N)
    num_train = len(xi_train_samples)
    
    def f_bar_grad_z(z, xi, current_theta):
        grad_f = loss_grad_z(current_theta, z)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    for epoch in range(N_EPOCHS):
        z_k = xi_train_samples.copy()
        for _ in range(K_INNER_STEPS):
            grad = f_bar_grad_z(z_k, xi_train_samples, theta)
            z_k += INNER_STEP_SIZE * grad
            z_k = np.clip(z_k, -1, 1)
            
        total_grad = loss_grad_theta(theta, z_k).sum(axis=0)
            
        avg_grad = total_grad / num_train
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- Dual Method Implementation (Vectorized) ---
def solve_dual(xi_train_samples):
    """ Solves Sinkhorn DRO using the dual method. """
    theta = np.zeros(DIM_N)
    num_train = len(xi_train_samples)
    
    levels = np.arange(SINKHORN_SAMPLE_LEVEL + 1)
    numerators = 2.0**(-levels)
    denominator = 2.0 - 2.0**(-SINKHORN_SAMPLE_LEVEL)
    probabilities = numerators / denominator
    
    for _ in range(N_EPOCHS):
        sampled_level = np.random.choice(levels, p=probabilities)
        m = 2**sampled_level
        
        noise = np.random.randn(num_train, m) * np.sqrt(EPSILON)
        z_samples = xi_train_samples[:, np.newaxis] + noise
        
        v = loss_function(theta, z_samples.flatten()).reshape(num_train, m) / (LAMBDA_VAL * EPSILON)
        
        v_max = np.max(v, axis=1, keepdims=True)
        weights = np.exp(v - v_max) 
        weights /= np.sum(weights, axis=1, keepdims=True)
        
        grads = loss_grad_theta(theta, z_samples.flatten()).reshape(num_train, m, -1)
        weighted_grads = np.sum(weights[:, :, np.newaxis] * grads, axis=1)
        total_grad = np.sum(weighted_grads, axis=0)

        avg_grad = total_grad / num_train
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- Experiment Execution and Evaluation ---
def run_experiment():
    """ Executes the complete experimental procedure. """
    xi_train = generate_training_data(N_TRAIN_SAMPLES)
    
    print("Solving ERM model...")
    theta_erm = solve_erm(xi_train)
    # print("Solving RGO model...")
    # theta_rgo = solve_rgo(xi_train)
    print("Solving WGF model...")
    theta_wgf = solve_wgf(xi_train)
    print("Solving WFR model...")
    theta_wfr = solve_wfr(xi_train)
    print("Solving WRM model...")
    theta_wrm = solve_wrm(xi_train)
    print("Solving Dual model...")
    theta_dual = solve_dual(xi_train)
    print("Solvinh MMD model...")
    theta_mmd_dro = solve_mmd_dro(xi_train)
    print("All models solved.")

    delta_values = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_STEPS)
    results = {'ERM': [], 'WGF': [], 'WFR': [], 'WRM': [], 'Dual': [], 'MMD': []}
    models = {
        'ERM': theta_erm,  'WGF': theta_wgf, 
        'WFR': theta_wfr, 'WRM': theta_wrm, 'Dual': theta_dual, 'MMD': theta_mmd_dro
    }

    print("Evaluating all models...")
    for delta in delta_values:
        xi_test = generate_test_data(N_TEST_SAMPLES, delta)
        for name, theta in models.items():
            test_loss = erm_objective_function(theta, xi_test)
            results[name].append(test_loss)
            
    print("Evaluation complete.")
    return delta_values, results

# --- Results Visualization ---
def plot_results(delta_values, results):
    """ Plots the loss curves for all models against the perturbation delta. """
    fig, ax = plt.subplots(figsize=(3.5, 2.613), dpi=300)

    styles = {
        'ERM': {'color': 'k', 'linestyle': '--'},
        'RGO': {'color': '#9467bd', 'linestyle': '-'}, 
        'WGF': {'color': '#d62728', 'linestyle': '-.'},
        'WFR': {'color': '#2ca02c', 'linestyle': ':'},
        'WRM': {'color': '#ff7f0e', 'linestyle': (0, (3, 1, 1, 1))}, # orange, dashdotdot
        'Dual': {'color': '#1f77b4', 'linestyle': (0, (5, 5))}, # blue, dashed
        'MMD': {'color': '#8c564b', 'linestyle': (0, (1, 1))} # brown, dotted
    }
    
    for name, losses in results.items():
        ax.plot(delta_values, losses, linewidth=1.5, label=name, **styles[name])

    #ax.set_title('(a) Uncertain least squares loss', fontsize=11)
    ax.set_xlabel(r'perturbation $\Delta$', fontsize=9)
    ax.set_ylabel('test loss', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.grid(False)
    ax.legend(fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig('uncertain_least_squares_results.png', dpi=300)
    plt.show()

# --- Main Program Entry ---
if __name__ == '__main__':
    with np.errstate(all='ignore'):
        delta_vals, all_results = run_experiment()
        plot_results(delta_vals, all_results)