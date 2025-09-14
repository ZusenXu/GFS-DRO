import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp

# --- 1. 实验设置与参数定义 ---
# 定义问题的维度
DIM_M = 10  # 矩阵 A 的行数
DIM_N = 10  # 矩阵 A 的列数 (theta 的维度)

# 定义训练/测试样本数量
N_TRAIN_SAMPLES = 20
N_TEST_SAMPLES = 1000

# 定义分布偏移的扰动范围
DELTA_MIN = 0.0
DELTA_MAX = 4.0
DELTA_STEPS = 50

# --- DRO 算法超参数 ---
N_EPOCHS = 50 # 外部优化的迭代轮数
LEARNING_RATE = 0.01 # 外部优化 theta 的学习率
LAMBDA_VAL = 0.5 # 正则化参数 lambda
EPSILON = 1 # 熵正则化参数 epsilon
GRAD_CLIP_THRESHOLD = 10.0 # 梯度裁剪阈值，防止梯度爆炸

# RGO (算法2) 特定参数
RGO_SMOOTHNESS_L = 0.5 # 假设的损失函数平滑度 L (需满足 LAMBDA_VAL > L/2)

# WGF (算法3), WFR (算法4), WRM 特定参数
K_INNER_STEPS = 50  # 内部循环迭代次数 K (现在作为基准次数)
INNER_STEP_SIZE = 0.01 # 内部循环步长 eta

# WFR (算法4) 特定参数
M_PARTICLES = 5 # WFR 使用的粒子(样本)数量
WFR_WEIGHT_STEP_SIZE = 0.08 # WFR 权重更新步长 tau

# Dual 方法特定参数
SINKHORN_SAMPLE_LEVEL = 4 # Dual 方法的蒙特卡洛采样级别

# --- 2. 生成固定的问题实例 ---
np.random.seed(42)
A0 = np.random.randn(DIM_M, DIM_N)
A1 = np.random.randn(DIM_M, DIM_N)
b = np.random.randn(DIM_M)

# --- 3. 数据生成函数 ---
def generate_training_data(n_samples):
    return np.random.uniform(-0.5, 0.5, n_samples)

def generate_test_data(n_samples, delta):
    lower_bound = -0.5 * (1 + delta)
    upper_bound = 0.5 * (1 + delta)
    return np.random.uniform(lower_bound, upper_bound, n_samples)

# --- 4. 核心函数定义 ---
def loss_function(theta, z):
    """ 计算单个样本 z 的损失 f_theta(z) """
    A_z = A0 + z * A1
    residual = A_z @ theta - b
    return np.sum(residual**2)/DIM_M

def loss_grad_theta(theta, z):
    """ 计算损失函数关于 theta 的梯度 nabla_theta f_theta(z) """
    A_z = A0 + z * A1
    residual = A_z @ theta - b
    return 2 * A_z.T @ residual

def loss_grad_z(theta, z):
    """ 计算损失函数关于 z 的梯度 nabla_z f_theta(z) """
    residual = (A0 + z * A1) @ theta - b
    grad_A_z = A1 @ theta
    return 2 * residual.T @ grad_A_z

# --- 5. ERM (经验风险最小化) 实现 ---
def erm_objective_function(theta, xi_samples):
    """ 计算 ERM 的目标函数值（平均损失）"""
    total_loss = np.mean([loss_function(theta, xi) for xi in xi_samples])
    return total_loss

def solve_erm(xi_train_samples):
    """ 使用优化器求解 ERM 问题 """
    initial_theta = np.zeros(DIM_N)
    result = minimize(
        erm_objective_function, initial_theta, args=(xi_train_samples,), method='BFGS')
    if not result.success:
        print("警告: ERM 优化过程可能未收敛。")
    return result.x

# --- 6. RGO (算法2) 实现 ---
def solve_rgo(xi_train_samples):
    """ 使用 RGO 算法求解 Sinkhorn DRO """
    theta = np.zeros(DIM_N)
    
    def rgo_inner_objective(z, xi, current_theta):
        l = loss_function(current_theta, z)
        penalty = np.sum((z - xi)**2)
        return -l / (LAMBDA_VAL * EPSILON) + penalty / EPSILON

    def rgo_inner_objective_grad_z(z, xi, current_theta):
        grad_l_z = loss_grad_z(current_theta, z)
        grad_penalty = 2 * (z - xi)
        return -grad_l_z / (LAMBDA_VAL * EPSILON) + grad_penalty / EPSILON
    
    for epoch in range(N_EPOCHS):
        grad_accumulator = np.zeros_like(theta)
        for xi in xi_train_samples:
            # Stage 1: 使用梯度下降寻找 z*
            steps = K_INNER_STEPS
            z_k = xi # 初始化
            for _ in range(steps):
                grad = rgo_inner_objective_grad_z(z_k, xi, theta)
                z_k -= INNER_STEP_SIZE * grad
            z_star = z_k

            # Stage 2: 拒绝采样
            accepted = False
            while not accepted:
                variance = (LAMBDA_VAL * EPSILON) / (2 * LAMBDA_VAL - RGO_SMOOTHNESS_L)
                if variance <= 0: variance = 1e-6
                z_candidate = np.random.normal(z_star, np.sqrt(variance))
                f_val_candidate = rgo_inner_objective(z_candidate, xi, theta)
                f_val_star = rgo_inner_objective(z_star, xi, theta)
                log_accept_prob = -f_val_candidate + f_val_star - ((np.sum((z_candidate - z_star)**2)) / (2*variance))
                if np.log(np.random.rand()) < log_accept_prob:
                    accepted_sample = z_candidate
                    accepted = True
            grad_accumulator += loss_grad_theta(theta, accepted_sample)
        
        avg_grad = grad_accumulator / len(xi_train_samples)
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- 7. Wasserstein Gradient Flow (WGF) 实现 (基于算法3) ---
def solve_wgf(xi_train_samples):
    """ 使用 Wasserstein Gradient Flow (WGF) 求解 Sinkhorn DRO """
    theta = np.zeros(DIM_N)
    
    def f_bar_grad_z(z, xi, current_theta):
        grad_f = loss_grad_z(current_theta, z)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    for epoch in range(N_EPOCHS):
        grad_accumulator = np.zeros_like(theta)
        for xi in xi_train_samples:
            z_k = xi
            steps = K_INNER_STEPS
            for _ in range(steps):
                noise = np.random.normal(0, 1)
                grad = f_bar_grad_z(z_k, xi, theta)
                z_k += INNER_STEP_SIZE * grad + np.sqrt(2 * INNER_STEP_SIZE * LAMBDA_VAL * EPSILON) * noise
            grad_accumulator += loss_grad_theta(theta, z_k)
            
        avg_grad = grad_accumulator / len(xi_train_samples)
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- 8. WFR Flow (算法4) 实现 ---
def solve_wfr(xi_train_samples):
    """ 使用 WFR Flow 求解 Sinkhorn DRO """
    theta = np.zeros(DIM_N)
    
    def f_bar(z, xi, current_theta):
         return loss_function(current_theta, z) - LAMBDA_VAL * np.sum((z-xi)**2)
    
    def f_bar_grad_z(z, xi, current_theta):
        grad_f = loss_grad_z(current_theta, z)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    for epoch in range(N_EPOCHS):
        grad_accumulator = np.zeros_like(theta)
        for xi in xi_train_samples:
            particles = np.full(M_PARTICLES, xi)
            weights = np.full(M_PARTICLES, 1.0 / M_PARTICLES)
            
            steps = K_INNER_STEPS
            for _ in range(steps):
                for i in range(M_PARTICLES):
                    noise = np.random.normal(0, 1)
                    grad = f_bar_grad_z(particles[i], xi, theta)
                    particles[i] += INNER_STEP_SIZE * grad + np.sqrt(2 * INNER_STEP_SIZE * LAMBDA_VAL * EPSILON) * noise
                    f_bar_val = f_bar(particles[i], xi, theta)
                    weights[i] = (weights[i]**(1 - LAMBDA_VAL * EPSILON * WFR_WEIGHT_STEP_SIZE)) * np.exp(WFR_WEIGHT_STEP_SIZE * f_bar_val)
                
                sum_weights = np.sum(weights)
                if sum_weights > 1e-8: weights /= sum_weights
                else: weights = np.full(M_PARTICLES, 1.0 / M_PARTICLES)

            weighted_grad = np.sum([w * loss_grad_theta(theta, z) for w, z in zip(weights, particles)], axis=0)
            grad_accumulator += weighted_grad
        
        avg_grad = grad_accumulator / len(xi_train_samples)
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- 9. WRM (Wasserstein Robust Method) 实现 ---
def solve_wrm(xi_train_samples):
    """ 使用 WRM (确定性内部优化) 求解 """
    theta = np.zeros(DIM_N)
    
    def f_bar_grad_z(z, xi, current_theta):
        grad_f = loss_grad_z(current_theta, z)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    for epoch in range(N_EPOCHS):
        grad_accumulator = np.zeros_like(theta)
        for xi in xi_train_samples:
            z_k = xi
            steps = K_INNER_STEPS
            for _ in range(steps):
                grad = f_bar_grad_z(z_k, xi, theta)
                z_k += INNER_STEP_SIZE * grad
            
            grad_accumulator += loss_grad_theta(theta, z_k)
            
        avg_grad = grad_accumulator / len(xi_train_samples)
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > GRAD_CLIP_THRESHOLD:
            avg_grad = avg_grad * GRAD_CLIP_THRESHOLD / grad_norm
        theta -= LEARNING_RATE * avg_grad
        
    return theta

# --- 10. Dual Method (对偶方法) 实现 ---
def solve_dual(xi_train_samples):
    """ 使用对偶方法求解 Sinkhorn DRO """
    theta = np.zeros(DIM_N)
    
    levels = np.arange(SINKHORN_SAMPLE_LEVEL + 1)
    numerators = 2.0**(-levels)
    denominator = 2.0 - 2.0**(-SINKHORN_SAMPLE_LEVEL)
    probabilities = numerators / denominator
    
    for _ in range(N_EPOCHS):
        grad_accumulator = np.zeros_like(theta)
        for xi in xi_train_samples:
            sampled_level = np.random.choice(levels, p=probabilities)
            m = 2**sampled_level
            noise = np.random.randn(m) * np.sqrt(EPSILON)
            z_samples = xi + noise
            v = np.array([loss_function(theta, z) for z in z_samples]) / (LAMBDA_VAL * EPSILON)
            weights = np.exp(v - np.max(v)) 
            weights /= np.sum(weights)
            grads_per_sample = np.array([loss_grad_theta(theta, z) for z in z_samples])
            weighted_grad = np.sum(weights[:, np.newaxis] * grads_per_sample, axis=0)
            grad_accumulator += weighted_grad

        avg_grad = grad_accumulator / len(xi_train_samples)
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
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
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

