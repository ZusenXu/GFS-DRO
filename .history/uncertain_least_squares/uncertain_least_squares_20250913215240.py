import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp
import os

# --- 1. 实验设置与参数定义 (已与官方文件对齐) ---
# 定义问题的维度
DIM_M = 20  # 矩阵 A 的行数 (已根据官方数据修正)
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
EPSILON = 0.5 # 熵正则化参数 epsilon
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

# --- 2. 加载官方数据集 (硬编码以移除 cvxopt 依赖) ---
print("正在加载官方数据集 (硬编码版本)...")
# 从 robls.bin 文件中提取的数据
A0_flat = [
    0.25003787, -0.05917881, -0.23893546, 0.80409824, 0.72834088,
    2.25948004, 0.87784937, -1.17812359, 1.17109974, 0.28661532,
    -1.42182999, -0.21927421, -0.15292762, 0.71754808, 0.10592679,
    -0.23814046, -1.10802367, 0.39683744, -1.1725309, 0.28749557,
    -0.50788541, -0.20682369, -0.70924466, 1.06761948, 0.60617808,
    1.66576254, 0.53493823, -1.17613997, -0.00731161, -0.81914582,
    0.96717266, 0.45028478, -0.50514112, -0.78287997, -0.45783088,
    0.25610097, -0.05563097, 0.06459789, -1.18546002, 0.50803566,
    0.63414537, 0.14602885, 0.1372974, -0.44577387, -1.22220436,
    -0.5683438, -1.09837733, 1.66535667, 0.73258043, -0.8716961,
    -0.19847685, 0.28349911, -0.77011756, 0.4852839, 0.58750813,
    -0.36741047, -0.6121577, 0.19685857, 0.46718984, -0.90855702,
    0.01242276, -0.11043147, 0.24304334, -1.32833193, -1.3675264,
    -2.25982067, -1.2640378, 2.28305282, -1.05578622, 0.07943218,
    1.05624088, 0.63339661, 0.14171268, -0.27125339, 0.31954098,
    0.47400092, 0.49104886, -0.97220237, 1.16475825, -0.35766948,
    -0.14129039, 0.34629778, 0.27283808, 0.19385463, 0.74489924,
    -0.80643975, 0.46229319, -0.72185528, -0.60700379, 0.39633883,
    -0.17306268, -0.85630984, 1.05281347, -0.20078964, -0.02762406,
    -0.2173559, 0.55858143, 0.45636177, 0.59690439, 0.44997981,
    -0.32914406, 0.63877355, -0.54978957, 1.27722309, 2.32632642,
    1.86954783, 1.43689268, -3.03534068, -0.11956323, 0.57975362,
    -0.69181718, -0.84664292, 1.20307884, 0.17007324, -0.49364612,
    -0.0645528, -0.22527475, 0.68055921, -0.82211579, 1.18396751,
    -0.01554297, -0.36476277, 0.2276415, -0.33453183, -0.43262069,
    -0.42015688, 0.09568789, 0.46282388, 0.00129392, 0.05269774,
    -0.05434922, 0.1463768, -0.19314829, -0.04309809, 0.1800566,
    0.0758013, 0.07863755, -0.29762305, 0.10745123, -0.05990911,
    -0.16819982, -0.8766473, 0.45053147, -0.55942491, -0.24564447,
    -1.34767835, 0.00254835, 1.08958615, -0.41981826, 1.21148646,
    -0.47824835, 0.19427199, 0.92158534, -0.03061487, 0.09289388,
    0.1792477, 0.1401624, -0.25336078, 1.226803, 0.29405587,
    -0.30898885, -1.1289769, 0.595893, 0.18832574, 0.60058916,
    0.16113351, 1.64854221, -0.94061951, 0.76110121, 1.14184298,
    -1.59859359, -0.34385834, 0.47790225, 0.06238379, 0.26433977,
    -0.242734, -0.35225751, 0.54663129, -0.1714633, 0.71002153,
    0.34016765, 1.09058028, -1.11635118, 1.70837408, 0.61271254,
    3.67914908, 0.0624467, -1.8808227, 1.39035472, -2.17182231,
    0.36928193, -0.01267536, -1.76752828, 0.33134577, -0.13910061,
    -0.36533788, -1.17162984, 0.8428494, -2.45355302, -0.42316362]
A0 = np.array(A0_flat).reshape(DIM_N, DIM_M).T

A1_flat = [
    -0.09393893, 0.04765061, 0.08725006, 0.02883281, -0.03527996,
    0.24096603, -0.08595977, 0.01464628, -0.09612727, 0.05623319,
    -0.2581709, 0.08827467, -0.14739316, -0.01376602, -0.14168357,
    -0.16558236, -0.25256966, -0.13015733, 0.02835751, 0.06596766,
    0.20684014, 0.08622027, -0.28721148, -0.13754497, -0.01892794,
    -0.33779225, -0.04184211, -0.07936001, 0.04442774, 0.07471207,
    0.12030433, -0.02685792, -0.03124774, 0.19383624, 0.24547024,
    -0.05119187, 0.02073282, -0.12896264, 0.0051813, 0.10233319,
    -0.07948119, 0.01741243, 0.00468091, -0.08351192, -0.07216189,
    0.25132917, -0.1342034, -0.06650293, -0.20112101, -0.0127605,
    0.05303795, -0.07046935, 0.04606528, -0.09669583, 0.16954322,
    -0.00417679, 0.10403817, 0.29150618, 0.05853078, 0.17825112,
    0.03356633, -0.04150648, -0.12339524, -0.0514402, 0.15483574,
    0.11530659, -0.13832853, -0.0257582, -0.17375835, 0.04098843,
    0.12954338, -0.09990791, -0.37379708, 0.09314519, 0.42872754,
    0.07165813, 0.03989949, 0.03797338, 0.12829636, 0.27908777,
    0.20383267, -0.02610418, 0.00230751, 0.03293295, -0.11423502,
    0.15478255, 0.02164525, 0.20816195, -0.02587464, 0.22076043,
    -0.25384533, 0.24365111, -0.20299506, -0.14051634, 0.00667515,
    -0.1944836, 0.06237194, 0.07291878, 0.17651184, -0.05407635,
    0.05594454, 0.03306775, 0.09413165, 0.08448313, -0.13266768,
    0.23618908, -0.05108297, -0.12184735, -0.13316605, 0.15977716,
    -0.02131638, 0.36149793, -0.02653721, -0.01890421, 0.05498191,
    -0.15762704, -0.03713547, -0.2113691, 0.14024251, 0.04046613,
    -0.1639696, 0.00930475, -0.05300977, 0.18566788, 0.18240795,
    0.08798762, 0.06568312, 0.10704911, -0.10257525, 0.0487188,
    -0.05319662, 0.11554448, -0.02374554, 0.25058364, 0.02304798,
    0.17001783, -0.00472515, -0.20071081, 0.04884632, -0.02655859,
    -0.18579019, 0.03065855, 0.03625509, -0.04897825, 0.18408415,
    -0.10662118, -0.12523171, -0.00032533, 0.05295227, -0.06800657,
    -0.0358668, -0.14467003, 0.26926707, 0.25989514, -0.20341141,
    0.25926511, -0.1197638, -0.04066245, 0.07423111, -0.04334816,
    0.17898445, 0.16340759, -0.21336437, -0.14274356, -0.10159267,
    0.05628148, -0.05793711, 0.09441796, 0.03046191, -0.1359604,
    0.01869776, -0.10001108, -0.08408689, 0.1864732, -0.22027781,
    -0.26938099, 0.06573703, -0.20788274, -0.01879786, 0.03274622,
    -0.0357562, -0.07623542, -0.02503547, 0.16157046, -0.0854077,
    0.08095394, -0.08348604, 0.07650048, -0.09684555, 0.15938316,
    0.02496594, 0.21322543, 0.04295515, 0.11588688, 0.17270499,
    0.05806273, -0.05163153, -0.01504851, -0.12892051, -0.27503139]
A1 = np.array(A1_flat).reshape(DIM_N, DIM_M).T

b_flat = [
    0.14999607, 0.54203757, 0.25440882, -0.30724069, -0.41711183,
    1.13680483, 0.39131381, 1.60514782, 0.82589231, 1.47039036,
    -1.37890689, -0.26017207, 0.99476817, 1.83403368, -1.71591032,
    0.08693171, 1.95567435, 0.16145377, -0.62868836, -1.43882447]
b = np.array(b_flat)

print("官方数据集加载成功。")

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
    loss = np.sum(residual**2, axis=1)/DIM_M
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

