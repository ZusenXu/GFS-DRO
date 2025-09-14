import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. 实验设置与参数定义 ---
# 定义问题的维度
DIM_M = 10  # 矩阵 A 的行数
DIM_N = 10  # 矩阵 A 的列数 (theta 的维度)

# 定义训练/测试样本数量
N_TRAIN_SAMPLES = 10
N_TEST_SAMPLES = 1000

# 定义分布偏移的扰动范围
DELTA_MIN = 0.0
DELTA_MAX = 4.0
DELTA_STEPS = 50

# --- DRO 算法超参数 ---
N_EPOCHS = 50 # 外部优化的迭代轮数
LEARNING_RATE = 0.01 # 外部优化 theta 的学习率
LAMBDA_VAL = 1.0 # 正则化参数 lambda
EPSILON = 0.01 # 熵正则化参数 epsilon

# RGO (算法2) 特定参数
RGO_SMOOTHNESS_L = 0.5 # 假设的损失函数平滑度 L (需满足 LAMBDA_VAL > L/2)

# MultiLD (算法3) 和 WFR (算法4) 特定参数
K_INNER_STEPS = 10  # 内部循环迭代次数 K
INNER_STEP_SIZE = 0.01 # 内部循环步长 eta

# WFR (算法4) 特定参数
M_PARTICLES = 5 # WFR 使用的粒子(样本)数量
WFR_WEIGHT_STEP_SIZE = 0.08 # WFR 权重更新步长 tau

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
    return np.sum(residual**2)

def loss_grad_theta(theta, z):
    """ 计算损失函数关于 theta 的梯度 nabla_theta f_theta(z) """
    A_z = A0 + z * A1
    residual = A_z @ theta - b
    return 2 * A_z.T @ residual

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
    
    # f_{lambda, xi}(z) = -f_theta(z)/(lambda*epsilon) + ||z-xi||^2/epsilon
    def rgo_inner_objective(z, xi, current_theta):
        l = loss_function(current_theta, z)
        penalty = np.sum((z - xi)**2)
        return -l / (LAMBDA_VAL * EPSILON) + penalty / EPSILON
    
    for _ in range(N_EPOCHS):
        grad_accumulator = np.zeros_like(theta)
        for xi in xi_train_samples:
            # Step 1: 找到 z*
            res = minimize(rgo_inner_objective, xi, args=(xi, theta), method='BFGS')
            z_star = res.x

            # Step 2: 拒绝采样
            accepted = False
            while not accepted:
                # 论文要求 lambda > L/2, 确保 2*lambda-L > 0
                variance = (LAMBDA_VAL * EPSILON) / (2 * LAMBDA_VAL - RGO_SMOOTHNESS_L)
                z_candidate = np.random.normal(z_star, np.sqrt(variance))
                
                f_val_candidate = rgo_inner_objective(z_candidate, xi, theta)
                f_val_star = rgo_inner_objective(z_star, xi, theta)
                
                # 计算接受概率
                log_accept_prob = -f_val_candidate + f_val_star - \
                                  ((np.sum((z_candidate - z_star)**2)) / (2*variance))
                
                if np.log(np.random.rand()) < log_accept_prob:
                    accepted_sample = z_candidate
                    accepted = True
            
            # 计算梯度并累加
            grad_accumulator += loss_grad_theta(theta, accepted_sample)
        
        # 更新 theta
        theta -= LEARNING_RATE * (grad_accumulator / len(xi_train_samples))
        
    return theta

# --- 7. Multi-Step Langevin Dynamics (算法3) 实现 ---
def solve_multi_ld(xi_train_samples):
    """ 使用 Multi-step Langevin Dynamics (MultiLD) 求解 Sinkhorn DRO """
    theta = np.zeros(DIM_N)
    
    # f_bar = f_theta(z) - lambda*||z-x||^2
    # grad_z f_bar = grad_z f_theta(z) - 2*lambda*(z-x)
    def f_bar_grad_z(z, xi, current_theta):
        # grad_z f_theta(z) = 2 * (A1*theta)^T * (A(z)*theta - b)
        A1_theta = A1 @ current_theta
        A_z_theta = (A0 + z * A1) @ current_theta
        grad_f = 2 * A1_theta.T @ (A_z_theta - b)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    for _ in range(N_EPOCHS):
        grad_accumulator = np.zeros_like(theta)
        for xi in xi_train_samples:
            z_k = xi  # 初始化
            # 内部 K 步迭代
            for _ in range(K_INNER_STEPS):
                noise = np.random.normal(0, 1)
                grad = f_bar_grad_z(z_k, xi, theta)
                z_k += INNER_STEP_SIZE * grad + \
                       np.sqrt(2 * INNER_STEP_SIZE * LAMBDA_VAL * EPSILON) * noise
            
            # 使用生成的对抗样本计算梯度
            grad_accumulator += loss_grad_theta(theta, z_k)
            
        # 更新 theta
        theta -= LEARNING_RATE * (grad_accumulator / len(xi_train_samples))
        
    return theta

# --- 8. WFR Flow (算法4) 实现 ---
def solve_wfr(xi_train_samples):
    """ 使用 WFR Flow 求解 Sinkhorn DRO """
    theta = np.zeros(DIM_N)
    
    def f_bar(z, xi, current_theta):
         return loss_function(current_theta, z) - LAMBDA_VAL * np.sum((z-xi)**2)
    
    # f_bar 的梯度与 MultiLD 中相同
    def f_bar_grad_z(z, xi, current_theta):
        A1_theta = A1 @ current_theta
        A_z_theta = (A0 + z * A1) @ current_theta
        grad_f = 2 * A1_theta.T @ (A_z_theta - b)
        return grad_f - 2 * LAMBDA_VAL * (z - xi)

    for _ in range(N_EPOCHS):
        grad_accumulator = np.zeros_like(theta)
        for xi in xi_train_samples:
            # 初始化 m 个粒子和权重
            particles = np.full(M_PARTICLES, xi)
            weights = np.full(M_PARTICLES, 1.0 / M_PARTICLES)
            
            # 内部 K 步迭代
            for _ in range(K_INNER_STEPS):
                for i in range(M_PARTICLES):
                    # 更新粒子位置
                    noise = np.random.normal(0, 1)
                    grad = f_bar_grad_z(particles[i], xi, theta)
                    particles[i] += INNER_STEP_SIZE * grad + \
                                    np.sqrt(2 * INNER_STEP_SIZE * LAMBDA_VAL * EPSILON) * noise
                    
                    # 更新权重
                    f_bar_val = f_bar(particles[i], xi, theta)
                    weights[i] = (weights[i]**(1 - LAMBDA_VAL * EPSILON * WFR_WEIGHT_STEP_SIZE)) * \
                                 np.exp(WFR_WEIGHT_STEP_SIZE * f_bar_val)
                
                # 归一化权重
                weights /= np.sum(weights)

            # 使用加权梯度
            weighted_grad = np.sum([w * loss_grad_theta(theta, z) for w, z in zip(weights, particles)], axis=0)
            grad_accumulator += weighted_grad
        
        # 更新 theta
        theta -= LEARNING_RATE * (grad_accumulator / len(xi_train_samples))
        
    return theta

# --- 9. 实验执行与评估 ---
def run_experiment():
    """ 执行完整的实验流程 """
    xi_train = generate_training_data(N_TRAIN_SAMPLES)
    
    print("正在求解 ERM 模型...")
    theta_erm = solve_erm(xi_train)
    print("正在求解 RGO 模型...")
    theta_rgo = solve_rgo(xi_train)
    print("正在求解 MultiLD 模型...")
    theta_multi_ld = solve_multi_ld(xi_train)
    print("正在求解 WFR 模型...")
    theta_wfr = solve_wfr(xi_train)
    print("所有模型求解完成。")

    delta_values = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_STEPS)
    results = {'ERM': [], 'RGO': [], 'MultiLD': [], 'WFR': []}
    models = {'ERM': theta_erm, 'RGO': theta_rgo, 'MultiLD': theta_multi_ld, 'WFR': theta_wfr}

    print("正在评估所有模型...")
    for delta in delta_values:
        xi_test = generate_test_data(N_TEST_SAMPLES, delta)
        for name, theta in models.items():
            test_loss = erm_objective_function(theta, xi_test)
            results[name].append(test_loss)
            
    print("评估完成。")
    return delta_values, results

# --- 10. 结果可视化 ---
def plot_results(delta_values, results):
    """ 绘制所有模型损失随扰动 delta 变化的曲线 """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    styles = {
        'ERM': {'color': 'k', 'linestyle': '--', 'label': 'ERM'},
        'RGO': {'color': '#9467bd', 'linestyle': '-', 'label': 'RGO'}, # purple
        'MultiLD': {'color': '#d62728', 'linestyle': '-.', 'label': 'MultiLD'}, # red
        'WFR': {'color': '#2ca02c', 'linestyle': ':', 'label': 'WFR'}, # green
    }
    
    for name, losses in results.items():
        ax.plot(delta_values, losses, linewidth=1.5, **styles[name])

    ax.set_title('(a) Uncertain least squares loss', fontsize=11)
    ax.set_xlabel('perturbation $\Delta$', fontsize=11)
    ax.set_ylabel('test loss', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.set_xlim(0, 4)
    ax.grid(False)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    delta_vals, all_results = run_experiment()
    plot_results(delta_vals, all_results)

