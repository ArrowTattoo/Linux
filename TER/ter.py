import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. 基础设置
np.random.seed(42)
true_params = np.array([1, 0, 2])
NB_Data = 50  # 为了演示这种方法，我们先把数据量设小一点，不然计算有点慢

def calculateY(x, params):
    a, b, c = params
    return a * x**2 + b * x + c

# 2. 生成拉普拉斯数据 (L1 的主场)
def generate_laplace_data(true_params, NB_Data):
    x = np.linspace(-2, 2, NB_Data)
    y_true = calculateY(x, true_params)
    # 拉普拉斯噪音：这种分布下，L1 比 L2 准得多
    noise = np.random.laplace(0, 0.5, size=len(x))
    return x, y_true + noise, y_true

# =======================================================
# 核心修改：L1 的 Pro 级解法 (转化为线性规划形式)
# =======================================================
def solve_l1_pro(x, y):
    """
    将 L1 回归转化为带约束的优化问题。
    变量: [a, b, c, t1, t2, ..., tn]
    """
    n = len(x)
    
    # 1. 初始猜测
    # 前3个是 a,b,c (随便猜 1,1,1)
    # 后 n 个是 t_i (也就是误差，暂时全设为 1)
    initial_params = np.concatenate([[1, 1, 1], np.ones(n)])
    
    # 2. 目标函数: 我们要最小化所有 t_i 的和
    # params[3:] 就是所有的 t 变量
    def objective(params):
        t_values = params[3:] 
        return np.sum(t_values)

    # 3. 约束条件: -t <= error <= t
    def constraint(params):
        a, b, c = params[:3]
        t_values = params[3:]
        
        # 算出当前的预测值
        y_pred = a * x**2 + b * x + c
        error = y - y_pred
        
        # 约束 1: t - error >= 0 (即 t >= error)
        con1 = t_values - error
        # 约束 2: t + error >= 0 (即 t >= -error)
        con2 = t_values + error
        
        # 把这两大堆约束拼起来
        return np.concatenate([con1, con2])

    # 4. 告诉求解器这是一个不等式约束 (ineq means >= 0)
    cons = {'type': 'ineq', 'fun': constraint}
    
    # 5. 调用 SLSQP
    # 这里的 options={'maxiter': 1000} 是给它多一点时间去算
    print(f"L1 Pro 正在求解 {len(initial_params)} 个变量的方程组，请稍等...")
    res = minimize(objective, initial_params, method='SLSQP', constraints=cons, options={'maxiter': 1000})
    
    if not res.success:
        print("警告: L1 优化未完全收敛，可能因为变量太多。")
    
    # 只返回前三个参数 a,b,c
    return res.x[:3]

# =======================================================
# 对照组：普通的 L2 解法 (BFGS)
# =======================================================
def solve_l2(x, y):
    def loss(params):
        y_guess = calculateY(x, params)
        return np.sum((y - y_guess)**2)
    res = minimize(loss, [1, 1, 1], method='BFGS')
    return res.x

# 3. 运行实验
fig, ax = plt.subplots(figsize=(10, 6))

# 生成数据
x, y, y_true = generate_laplace_data(true_params, NB_Data)

# 计算 L1 (用 Pro 方法)
params_l1 = solve_l1_pro(x, y)

# 计算 L2 (用普通方法)
params_l2 = solve_l2(x, y)

# 计算误差
e_l1 = np.sum(np.abs(true_params - params_l1))
e_l2 = np.sum(np.abs(true_params - params_l2))

# 绘图
ax.scatter(x, y, color='gray', alpha=0.5, label='Data (Laplace Noise)')
ax.plot(x, y_true, 'k--', linewidth=2, label='True Truth')
ax.plot(x, calculateY(x, params_l1), 'b-', linewidth=2, label=f'L1 Pro Fit (Err={e_l1:.2f})')
ax.plot(x, calculateY(x, params_l2), 'r--', linewidth=2, label=f'L2 BFGS Fit (Err={e_l2:.2f})')

ax.set_title(f"L1 Optimization: Pro Method (Linear Programming Logic)\nData Points: {NB_Data}")
ax.legend()
ax.grid(True, alpha=0.3)

print(f"\n真实参数: {true_params}")
print(f"L1 Pro 算出的参数: {params_l1} (误差: {e_l1:.4f})")
print(f"L2 BFGS 算出的参数: {params_l2} (误差: {e_l2:.4f})")

plt.show()
