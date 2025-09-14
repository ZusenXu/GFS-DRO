import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp
import pickle
import os

# --- 1. 实验设置与参数定义 (已与官方文件对齐) ---
# 定义问题的维度
DIM_M = 10  # 矩阵 A 的行数
DIM_N = 10  # 矩阵 A 的列数 (theta 的维度)

# 定义训练/测试样本数量
N_TRAIN_SAMPLES = 10 # 官方: n_sample = 10
N_TEST_SAMPLES = 500 # 官方: n_test = 500

# 定义分布偏移的扰动范围
DELTA_MIN = 0.0
DELTA_MAX = 5.0 # 官方: disturb_set goes up to 5.0
DELTA_STEPS = 50

# --- DRO 算法超参数 ---
N_EPOCHS = 50 # 外部优化的迭代轮数
LEARNING_RATE = 0.01 # 外部优化 theta 的学习率
LAMBDA_VAL = 1 # 正则化参数 lambda
EPSILON = 0.1 # 熵正则化参数 epsilon
GRAD_CLIP_THRESHOLD = 10.0 # 梯度裁剪阈值，防止梯度爆炸

# RGO (算法2) 特定参数
RGO_SMOOTHNESS_L = 0.5 # 假设的损失函数平滑度 L (需满足 LAMBDA_VAL > L/2)

# WGF (算法3), WFR (算法4), WRM 特定参数
K_INNER_STEPS = 50  # 内部循环迭代次数 K
INNER_STEP_SIZE = 0.01 # 内部循环步长 eta

# WGF, WFR, RGO 特定参数
M_PARTICLES = 5 # 使用的粒子(样本)数量

# WFR (算法4) 特定参数
WFR_WEIGHT_STEP_SIZE = 0.08 # WFR 中权重更新的步长

# Dual 方法特定参数
SINKHORN_SAMPLE_LEVEL = 4 # Dual 方法的蒙特卡洛采样级别

# --- 2. 生成固定的问题实例 (已更新为从 robls.bin 加载) ---
try:
    # 尝试从官方文件加载数据
    data_path = 'robls.bin'
    if os.path.exists(data_path):
        print(f"正在从 '{data_path}' 加载官方数据集...")
        datafile = pickle.load(open(data_path, 'rb'))['6.15']
        # 将 cvxopt.matrix 转换为 numpy array
        A0 = np.array(datafile['A']).T
        B_matrix = np.array(datafile['B']).T
        # 官方代码中 B 是一个列表，我们这里只取第一个元素 A1
        A1 = B_matrix[:, :DIM_N]
        b = np.array(datafile['b']).flatten()
        print("官方数据集加载成功。")
    else:
        raise FileNotFoundError

except (FileNotFoundError, KeyError):
    print(f"警告: 未找到官方数据集 '{data_path}' 或文件格式不符。")
    print("将回退到随机生成数据。")
    np.random.seed(42)
    A0 = np.random.randn(DIM_M, DIM_N)
    A1 = np.random.randn(DIM_M, DIM_N)
    b = np.random.randn(DIM_M)

# --- 3. 数据生成函数 ---
def generate_training_data(n_samples):
    # 官方代码使用 U[-0.5, 0.5] 作为基础训练分布
    return np.random.uniform(-0.5, 0.5, n_samples)

def generate_test_data(n_samples, delta):
    # 官方代码使用 U[-0.5*(1+delta), 0.5*(1+delta)] 作为测试分布
    lower_bound = -0.5 * (1 + delta)
    upper_bound = 0.5 * (1 + delta)
    return np.random.uniform(lower_bound, upper_bound, n_samples)

# --- 4. 核心函数定义 (Vectorized & Loss Corrected) ---
def loss_function(theta, z):
    """ 计算单个样本或一批样本 z 的损失 f_theta(z) (平方和) """
    z = np.atleast_1d(z)
    A_z = A0[np.newaxis, :, :] + z[:, np.newaxis, np.newaxis] * A1[np.newaxis, :, :]
    residual = A_z @ theta - b[np.newaxis, :]
    # 已修正: 使用平方和 (sum of squares), 而非均方误差 (mean squared error)
    loss = np.sum(residual**2, axis=1)
    return loss.squeeze()

def loss_grad_theta(theta, z):
    """ 计算损失函数关于 theta 的梯度 (对应平方和损失) """
    z = np.atleast_1d(z)
    A_z = A0[np.newaxis, :, :] + z[:, np.newaxis, np.newaxis] * A1[np.newaxis, :, :]
    residual = A_z @ theta - b[np.newaxis, :]
    grad = 2 * (A_z.transpose(0, 2, 1) @ residual[:, :, np.newaxis]).squeeze(axis=2)
    return grad.squeeze()

def loss_grad_z(theta, z):
    """ 计算损失函数关于 z 的梯度 (对应平方和损失) """
    z = np.atleast_1d(z)
    A_z = A0[np.newaxis, :, :] + z[:, np.newaxis, np.newaxis] * A1[np.newaxis, :, :]
    residual = A_z @ theta - b[np.newaxis, :]
    grad_A_z = A1 @ theta
    grad = 2 * np.sum(residual * grad_A_z[np.newaxis, :], axis=1)
    return grad.squeeze()

# --- 5. ERM (经验风险最小化) 实现 ---
def erm_objective_function(theta, xi_samples):
    """ 计算 ERM 的目标函数值（平均损失）"""
    return np.mean(loss_function(theta, xi_samples))

def solve_erm(xi_train_samples):
    """ 使用优化器求解 ERM 问题 """
    initial_theta = np.zeros(DIM_N)
    result = minimize(
        erm_objective_function, initial_theta, args=(xi_train_samples,), method='BFGS')
    if not result.success:
        print("警告: ERM 优化过程可能未收敛。")
    return result.x

# --- 6. RGO (算法2) 实现 (Vectorized) ---
def solve_rgo(xi_train_samples):
    """ 使用 RGO 算法求解 Sinkhorn DRO """
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
        z_k = xi_train_samples.copy()
        for _ in range(K_INNER_STEPS):
            grad = rgo_inner_objective_grad_z(z_k, xi_train_samples, theta)
            z_k -= INNER_STEP_SIZE * grad
        z_star_batch = z_k

        variance = (LAMBDA_VAL * EPSILON) / (2 * LAMBDA_VAL - RGO_SMOOTHNESS_L)
        if variance <= 0: variance = 1e-6
        std_dev = np.sqrt(variance)
        
        z_star_expanded = z_star_batch[:, np.newaxis]
        xi_expanded = xi_train_samples[:, np.newaxis]
        
        final_accepted_samples = np.zeros((num_train, M_PARTICLES))
        active_flags = np.ones((num_train, M_PARTICLES), dtype=bool)
        
        for _ in range(100):
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

# --- 7. Wasserstein Gradient Flow (WGF) 实现 (Vectorized) ---
def solve_wgf(xi_train_samples):
    """ 使用 Wasserstein Gradient Flow (WGF) 求解 Sinkhorn DRO """
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

        all_grads = loss_grad_theta(theta, particles.flatten())
        avg_particle_grads = all_grads.reshape(num_train, M_PARTICLES, -1).mean(axis=1)
        total_grad = avg_particle_grads.sum(axis=0)
            
        avg_grad = total_grad / num_train
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- 8. WFR Flow (算法4) 实现 (Vectorized) ---
def solve_wfr(xi_train_samples):
    """ 使用 WFR Flow 求解 Sinkhorn DRO """
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

# --- 9. WRM (Wasserstein Robust Method) 实现 (Vectorized) ---
def solve_wrm(xi_train_samples):
    """ 使用 WRM (确定性内部优化) 求解 """
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
        total_grad = loss_grad_theta(theta, z_k).sum(axis=0)
        avg_grad = total_grad / num_train
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- 10. Dual Method (对偶方法) 实现 (Vectorized) ---
def solve_dual(xi_train_samples):
    """ 使用对偶方法求解 Sinkhorn DRO """
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

# --- 11. 实验执行与评估 ---
def run_experiment():
    """ 执行完整的实验流程 """
    xi_train = generate_training_data(N_TRAIN_SAMPLES)
    
    print("正在求解 ERM 模型...")
    theta_erm = solve_erm(xi_train)
    print("正在求解 RGO 模型...")
    theta_rgo = solve_rgo(xi_train)
    print("正在求解 WGF 模型...")
    theta_wgf = solve_wgf(xi_train)
    print("正在求解 WFR 模型...")
    theta_wfr = solve_wfr(xi_train)
    print("正在求解 WRM 模型...")
    theta_wrm = solve_wrm(xi_train)
    print("正在求解 Dual 模型...")
    theta_dual = solve_dual(xi_train)
    print("所有模型求解完成。")

    delta_values = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_STEPS)
    results = {'ERM': [], 'RGO': [], 'WGF': [], 'WFR': [], 'WRM': [], 'Dual': []}
    models = {
        'ERM': theta_erm, 'RGO': theta_rgo, 'WGF': theta_wgf, 
        'WFR': theta_wfr, 'WRM': theta_wrm, 'Dual': theta_dual
    }

    print("正在评估所有模型...")
    for delta in delta_values:
        xi_test = generate_test_data(N_TEST_SAMPLES, delta)
        for name, theta in models.items():
            test_loss = erm_objective_function(theta, xi_test)
            results[name].append(test_loss)
            
    print("评估完成。")
    return delta_values, results

# --- 12. 结果可视化 ---
def plot_results(delta_values, results):
    """ 绘制所有模型损失随扰动 delta 变化的曲线 """
    fig, ax = plt.subplots(figsize=(3.5, 2.613), dpi=300)

    styles = {
        'ERM': {'color': 'k', 'linestyle': '--'},
        'RGO': {'color': '#9467bd', 'linestyle': '-'}, 
        'WGF': {'color': '#d62728', 'linestyle': '-.'},
        'WFR': {'color': '#2ca02c', 'linestyle': ':'},
        'WRM': {'color': '#ff7f0e', 'linestyle': (0, (3, 1, 1, 1))}, # orange, dashdotdot
        'Dual': {'color': '#1f77b4', 'linestyle': (0, (5, 5))} # blue, dashed
    }
    
    for name, losses in results.items():
        ax.plot(delta_values, losses, linewidth=1.5, label=name, **styles[name])

    ax.set_title('(a) Uncertain least squares loss', fontsize=11)
    ax.set_xlabel('perturbation $\Delta$', fontsize=11)
    ax.set_ylabel('test loss', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.set_xlim(0, 5.0) # 已更新X轴范围
    ax.set_ylim(bottom=0) # Y轴从0开始
    ax.grid(False)
    ax.legend(fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig('uncertain_least_squares_results.png', dpi=300)
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    with np.errstate(all='ignore'):
        delta_vals, all_results = run_experiment()
        plot_results(delta_vals, all_results)

