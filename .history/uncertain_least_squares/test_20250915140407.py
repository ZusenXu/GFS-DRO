import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Experiment Setup and Parameter Definitions ---
DIM_M = 10
DIM_N = 10
N_TRAIN_SAMPLES = 20
LAMBDA_VAL = 0.1
K_INNER_STEPS = 3000
INNER_STEP_SIZE = 0.0001
M_PARTICLES = 8
EPSILON = 0.5

# --- Generate a Fixed Problem Instance ---
# Use a fixed random seed for reproducibility
np.random.seed(219)
A0 = np.random.randn(DIM_M, DIM_N)
A1 = np.random.randn(DIM_M, DIM_N)
b = np.random.randn(DIM_M)

# --- Data Generation Functions ---
def generate_training_data(n_samples):
    """Generate training data"""
    return np.random.uniform(-0.5, 0.5, n_samples)

# --- Core Function Definitions (Vectorized) ---
def loss_function(theta, z):
    """Computes the loss function f_theta(z) for a single sample or a batch of samples z."""
    z = np.atleast_1d(z)
    A_z = A0[np.newaxis, :, :] + z[:, np.newaxis, np.newaxis] * A1[np.newaxis, :, :]
    residual = A_z @ theta - b[np.newaxis, :]
    loss = np.sum(residual**2, axis=1) / DIM_M
    return loss.squeeze()

def loss_grad_z(theta, z):
    """Computes the gradient of the loss function with respect to z, nabla_z f_theta(z)"""
    z = np.atleast_1d(z)
    A_z = A0[np.newaxis, :, :] + z[:, np.newaxis, np.newaxis] * A1[np.newaxis, :, :]
    residual = A_z @ theta - b[np.newaxis, :]
    grad_A_z = A1 @ theta
    grad = 2 * np.sum(residual * grad_A_z[np.newaxis, :], axis=1)
    return grad.squeeze()

# --- ERM (Empirical Risk Minimization) Implementation ---
def erm_objective_function(theta, xi_samples):
    """Calculates the ERM objective function value (average loss)."""
    return np.mean(loss_function(theta, xi_samples))

def solve_erm(xi_train_samples):
    """Solves the ERM problem using an optimizer."""
    initial_theta = np.zeros(DIM_N)
    result = minimize(
        erm_objective_function, initial_theta, args=(xi_train_samples,), method='BFGS')
    if not result.success:
        print("Warning: ERM optimization may not have converged.")
    return result.x

# --- Function to get WGF final particle distribution ---
def get_wgf_final_particles(xi_train_samples, theta):
    """Runs the WGF inner loop for a fixed theta to get the final particles."""
    num_train = len(xi_train_samples)
    # Initialize particles, M_PARTICLES for each initial training sample
    particles = np.tile(xi_train_samples[:, np.newaxis], (1, M_PARTICLES))

    def f_bar_grad_z(z, xi, current_theta):
        grad_f = loss_grad_z(current_theta, z)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    # Run the inner loop to update particle positions
    for _ in range(K_INNER_STEPS):
        particles_flat = particles.flatten()
        xi_expanded = np.repeat(xi_train_samples, M_PARTICLES)
        
        grad = f_bar_grad_z(particles_flat, xi_expanded, theta)
        noise = np.random.normal(0, 1, size=particles_flat.shape)
        
        updated_particles_flat = particles_flat + INNER_STEP_SIZE * grad + np.sqrt(2 * INNER_STEP_SIZE * LAMBDA_VAL * EPSILON) * noise
        particles = updated_particles_flat.reshape(num_train, M_PARTICLES)
        particles = np.clip(particles, -1, 1) # Clip particles to be within the [-1, 1] range
        
    return particles.flatten()

# --- Main Execution Flow ---
if __name__ == '__main__':
    # 1. Generate training data
    xi_train = generate_training_data(N_TRAIN_SAMPLES)

    # 2. Solve for theta_erm
    print("Solving ERM model...")
    theta_erm = solve_erm(xi_train)
    print("ERM model solved.")

    # 3. Get the final particle distribution using the WGF method
    print("Calculating WGF final particle distribution...")
    final_particles_wgf = get_wgf_final_particles(xi_train, theta_erm)
    print("Particle distribution calculated.")

    # 4. Generate data required for plotting
    # xi range for plotting the curves
    xi_range = np.linspace(-1, 1, 400)
    
    # Calculate the black curve (loss)
    loss_values = loss_function(theta_erm, xi_range)
    
    # Calculate the penalty term for the red curve
    penalty_term = np.sum([LAMBDA_VAL * np.abs(xi_range - xi_i) for xi_i in xi_train], axis=0)
    red_curve_values = loss_values - penalty_term

    # --- Plotting ---
    print("Generating plot...")
    plt.figure(figsize=(4, 4)) # Set a square figure size

    # Plot the black and red curves
    plt.plot(xi_range, loss_values, 'k-', label='loss(ξ, θ_erm)')
    plt.plot(xi_range, red_curve_values, 'r-', label='loss(ξ, θ_erm) - Σ λ|ξ - ξ_i|')

    # Plot the initial training data (blue histogram)
    # density=True means the total area of the histogram is 1
    plt.hist(xi_train, bins=20, density=True, color='blue', alpha=0.6, label='Initial ξ_i')

    # Plot the final particle distribution from WGF (light red/magenta histogram)
    plt.hist(final_particles_wgf, bins=20, density=True, color='magenta', alpha=0.5, label='Final ξ_i (WGF)')

    # Mark a point on the red curve at xi=0
    xi_zero_index = np.argmin(np.abs(xi_range - 0))
    plt.plot(0, red_curve_values[xi_zero_index], 'mo', markersize=5)


    # Add vertical dashed boundary lines
    plt.axvline(x=-1, color='k', linestyle='--')
    plt.axvline(x=1, color='k', linestyle='--')

    # Set labels and axis limits
    plt.xlabel('ξ')
    plt.ylabel('loss')
    plt.ylim(0, 2)
    plt.xlim(-1.1, 1.1)
    plt.legend()
    plt.grid(False) # Do not display grid
    
    # Save and show the plot
    plt.savefig('uncertainty_visualization.png', dpi=300)
    plt.show()
    print("Plot generated and saved as uncertainty_visualization.png.")

