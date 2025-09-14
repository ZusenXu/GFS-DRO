import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. 实验设置与参数定义 ---
# 定义问题的维度
DIM_M = 10  # 矩阵 A 的行数
DIM_N = 10  # 矩阵 A 的列数 (theta 的维度)

# 定义训练样本数量
N_TRAIN_SAMPLES = 10

# 定义测试样本数量
N_TEST_SAMPLES = 1000

# 定义分布偏移的扰动范围
# Delta 将从 0 变化到 4
DELTA_MIN = 0.0
DELTA_MAX = 4.0
DELTA_STEPS = 50

# --- 2. 生成固定的问题实例 ---
# 为了实验的可复现性，我们设置一个随机种子
np.random.seed(42)

# 根据论文，A0, A1, b 是给定的。我们在这里随机生成一次并保持不变。
# A(xi) = A0 + xi * A1
A0 = np.random.randn(DIM_M, DIM_N)
A1 = np.random.randn(DIM_M, DIM_N)
b = np.random.randn(DIM_M)

# --- 3. 数据生成函数 ---

def generate_training_data(n_samples):
    """
    生成训练数据 xi，均匀分布在 [-0.5, 0.5]
    参数:
        n_samples (int): 生成的样本数量
    返回:
        numpy.ndarray: 包含 xi 样本的数组
    """
    return np.random.uniform(-0.5, 0.5, n_samples)

def generate_test_data(n_samples, delta):
    """
    生成有分布偏移的测试数据 xi
    xi 均匀分布在 [-0.5*(1+delta), 0.5*(1+delta)]
    参数:
        n_samples (int): 生成的样本数量
        delta (float): 分布偏移的扰动参数
    返回:
        numpy.ndarray: 包含测试 xi 样本的数组
    """
    lower_bound = -0.5 * (1 + delta)
    upper_bound = 0.5 * (1 + delta)
    return np.random.uniform(lower_bound, upper_bound, n_samples)

# --- 4. 经验风险最小化 (ERM) 实现 ---

def erm_objective_function(theta, xi_samples, A0, A1, b):
    """
    计算 ERM 的目标函数值（平均平方损失）
    目标: 最小化 1/N * sum(||A(xi) * theta - b||^2)
    参数:
        theta (numpy.ndarray): 模型参数
        xi_samples (numpy.ndarray): 训练数据中的 xi 值
        A0, A1, b: 问题定义的矩阵和向量
    返回:
        float: 所有样本的平均平方损失
    """
    total_loss = 0
    num_samples = len(xi_samples)
    
    for xi in xi_samples:
        A_xi = A0 + xi * A1
        residual = A_xi @ theta - b
        loss = np.sum(residual**2) # ||.||^2
        total_loss += loss
        
    return total_loss / num_samples

def solve_erm(xi_train_samples, A0, A1, b):
    """
    使用优化器求解 ERM 问题，找到最优的 theta
    参数:
        xi_train_samples (numpy.ndarray): 训练数据
        A0, A1, b: 问题定义的矩阵和向量
    返回:
        numpy.ndarray: 求解得到的最优 theta_erm
    """
    # 初始猜测 theta 为零向量
    initial_theta = np.zeros(DIM_N)
    
    # 使用 scipy.optimize.minimize 进行优化
    result = minimize(
        erm_objective_function, 
        initial_theta, 
        args=(xi_train_samples, A0, A1, b),
        method='BFGS' # 一种常用的优化算法
    )
    
    if not result.success:
        print("警告: ERM 优化过程可能未收敛。")
        
    return result.x

# --- 5. 实验执行与评估 ---

def run_experiment():
    """
    执行完整的实验流程
    """
    # 1. 生成训练数据
    xi_train = generate_training_data(N_TRAIN_SAMPLES)
    
    # 2. 求解 ERM 模型参数 theta_erm
    print("正在求解 ERM 模型参数 theta...")
    theta_erm = solve_erm(xi_train, A0, A1, b)
    print("ERM 模型参数求解完成。")
    
    # 3. 在不同扰动下进行测试
    delta_values = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_STEPS)
    erm_test_losses = []
    
    print("正在测试 ERM 模型在不同分布偏移下的性能...")
    for delta in delta_values:
        # 生成带偏移的测试数据
        xi_test = generate_test_data(N_TEST_SAMPLES, delta)
        
        # 在测试集上计算 ERM 模型的损失
        test_loss = erm_objective_function(theta_erm, xi_test, A0, A1, b)
        erm_test_losses.append(test_loss)
        
    print("测试完成。")
    return delta_values, erm_test_losses

# --- 6. 结果可视化 ---

def plot_results(delta_values, erm_losses):
    """
    绘制 ERM 损失随扰动 delta 变化的曲线, 并应用指定的论文绘图风格
    """
    # 创建图表，使用指定的尺寸和 DPI
    fig, ax = plt.subplots(figsize=(2.613, 2.613), dpi=300)

    # 绘制数据，使用指定的线宽 (1.5) 和样式 (黑色虚线)
    ax.plot(delta_values, erm_losses, 'k--', linewidth=1.5, label='ERM')

    # 设置标题和标签，使用指定的字体大小 (11)
    ax.set_title('(a) Uncertain least squares loss', fontsize=11)
    ax.set_xlabel('perturbation $\Delta$', fontsize=11)
    ax.set_ylabel('test loss', fontsize=11)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=11)

    # 设置坐标轴线宽
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    # 根据参考图片设置坐标轴范围
    ax.set_xlim(0, 4)
    # ax.set_ylim(0, 3) # 移除固定的 Y 轴范围，让 matplotlib 自动调整

    # 移除网格
    ax.grid(False)

    # 显示图例
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('uncertain_least_squares_loss.png', dpi=300)
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    delta_vals, erm_losses = run_experiment()
    plot_results(delta_vals, erm_losses)

