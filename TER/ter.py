import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

np.random.seed(42)       
TRUE_PARAMS = np.array([1, 0, 2]) # a=1, b=0, c=2
NB_DATA = 40             #
NB_EXPERIMENTS = 100     
initial = [1,1,1]

def calculateY(x, params):
    a, b, c = params
    return a * x**2 + b * x + c


def generate_data(dist_type, n_samples):
    x = np.linspace(-2, 2, n_samples)
    y_true = calculateY(x, TRUE_PARAMS)
    
    if dist_type == 'Normal':
        # 正态分布：L2 的主场
        noise = np.random.normal(0, 1.5, size=n_samples)
    elif dist_type == 'Uniform':
        # 均匀分布：L_inf 的主场
        noise = np.random.uniform(-2.5, 2.5, size=n_samples)
    elif dist_type == 'Laplace':
        # 拉普拉斯分布：L1 的主场
        noise = np.random.laplace(0, 0.5, size=n_samples)
        
    return x, y_true + noise


def generate_normal_data(true_params,NB_Data):
    x = np.linspace(-2, 2, NB_Data)
    y_true=calculateY(x,true_params)
    noise=np.random.normal(0,1.5,size=len(x))
    y=y_true+noise
    return x,y,y_true


def generate_uniform_data(true_params,NB_Data):
    x = np.linspace(-2, 2, NB_Data)
    y_true=calculateY(x,true_params)
    noise=np.random.uniform(-2.5,2.5,size=len(x))
    y=y_true+noise
    return x,y,y_true

def generate_laplace_data(true_params):
    x = np.linspace(-2, 2, 50)
    y_true = calculateY(x, true_params)
    noise = np.random.laplace(0, 0.5, size=len(x))
    y = y_true + noise
    return x, y, y_true

def loss_l2(params,x,y):
    y_guess=calculateY(x,params)
    return np.sum((y-y_guess)**2)

def solve_l2(x, y):
    res = minimize(loss_l2, initial, method='BFGS')
    return res.x

# --- L_inf 求解器 (SLSQP - 转化法) ---
def solve_linf(x, y):
    # 目标: min t
    # 约束: -t <= error <= t
    def objective(p): return p[3]
    def constraint(p):
        err = y - calculateY(x, p[:3])
        t = p[3]
        return np.concatenate([t - err, t + err])
    
    x0 = [1, 1, 1, 5] # 初始猜测
    cons = {'type': 'ineq', 'fun': constraint}
    # SLSQP 处理约束问题
    res = minimize(objective, x0, method='SLSQP', constraints=cons)
    return res.x[:3]

# --- L1 求解器 (SLSQP - 转化法 - 老师推荐的Pro版) ---
def solve_l1(x, y):
    # 目标: min sum(t_i)
    # 约束: -t_i <= error_i <= t_i
    n = len(x)
    # 参数结构: [a, b, c, t1, t2, ... tn]
    x0 = np.concatenate([[1, 1, 1], np.ones(n)])
    
    def objective(p): 
        return np.sum(p[3:]) # 求 t 的和
    
    def constraint(p):
        a, b, c = p[:3]
        t_s = p[3:]
        err = y - (a * x**2 + b * x + c)
        return np.concatenate([t_s - err, t_s + err])
    
    cons = {'type': 'ineq', 'fun': constraint}
    # 增加 maxiter 防止还没算完就停了
    res = minimize(objective, x0, method='SLSQP', constraints=cons, options={'maxiter': 1000})
    return res.x[:3]

# ==========================================
# 4. 实验核心逻辑
# ==========================================
def run_simulation():
    # 用于存储 100 次实验的结果
    # 结构: results['Normal']['L1'] = [第一次误差, 第二次误差, ...]
    distributions = ['Normal', 'Uniform', 'Laplace']
    solvers = ['L1', 'L2', 'L_inf']
    
    # 初始化记录本
    history = {dist: {solver: [] for solver in solvers} for dist in distributions}
    
    print(f"=== 开始 100 次蒙特卡洛模拟 (N={NB_DATA}) ===")
    print("正在计算中，Pro 方法比较复杂，请耐心等待...")
    
    start_time = time.time()
    
    for i in range(NB_EXPERIMENTS):
        # 打印进度条
        if (i+1) % 10 == 0:
            print(f"进度: {i+1}/{NB_EXPERIMENTS} 次实验...")

        for dist_name in distributions:
            # 1. 生成数据
            x, y = generate_data(dist_name, NB_DATA)
            
            # 2. 分别用三种方法求解
            # 这里我们把三种方法都跑一遍，看看谁在当前分布下最准
            
            # 求解 L1 (Pro)
            p_l1 = solve_l1(x, y)
            err_l1 = np.sum(np.abs(TRUE_PARAMS - p_l1))
            
            # 求解 L2 (BFGS)
            p_l2 = solve_l2(x, y)
            err_l2 = np.sum(np.abs(TRUE_PARAMS - p_l2))
            
            # 求解 L_inf (Pro)
            p_linf = solve_linf(x, y)
            err_linf = np.sum(np.abs(TRUE_PARAMS - p_linf))
            
            # 3. 记录成绩
            history[dist_name]['L1'].append(err_l1)
            history[dist_name]['L2'].append(err_l2)
            history[dist_name]['L_inf'].append(err_linf)
            
    total_time = time.time() - start_time
    print(f"\n计算完成！总耗时: {total_time:.2f} 秒")
    return history

# ==========================================
# 5. 绘图与分析
# ==========================================
def plot_results(history):
    # 计算平均误差
    distributions = ['Normal', 'Uniform', 'Laplace']
    solvers = ['L1', 'L2', 'L_inf']
    colors = ['blue', 'red', 'green']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        
        # 提取该分布下，三种方法的平均误差
        means = []
        stds = [] # 标准差，看波动的
        
        for solver in solvers:
            data = history[dist][solver]
            means.append(np.mean(data))
            stds.append(np.std(data))
            
        # 绘制柱状图
        bars = ax.bar(solvers, means, yerr=stds, capsize=10, color=colors, alpha=0.7)
        
        # 找出冠军
        best_solver_idx = np.argmin(means)
        best_solver_name = solvers[best_solver_idx]
        
        # 装饰图表
        ax.set_title(f"Environment: {dist}\nWinner: {best_solver_name}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Average Parameter Error (lower is better)")
        ax.grid(axis='y', alpha=0.3)
        
        # 在柱子上标数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # 打印最终结论表格
    print("\n" + "="*50)
    print(f"{'Distribution':<15} | {'L1 Error':<10} | {'L2 Error':<10} | {'L_inf Error':<10}")
    print("-" * 50)
    for dist in distributions:
        l1_err = np.mean(history[dist]['L1'])
        l2_err = np.mean(history[dist]['L2'])
        linf_err = np.mean(history[dist]['L_inf'])
        
        # 标记冠军
        best = min(l1_err, l2_err, linf_err)
        
        def mark(val): return f"*{val:.3f}*" if val == best else f"{val:.3f}"
        
        print(f"{dist:<15} | {mark(l1_err):<10} | {mark(l2_err):<10} | {mark(linf_err):<10}")
    print("="*50)
    print("(* 代表该环境下的冠军，误差最小)")

# 运行主程序
if __name__ == "__main__":
    history_data = run_simulation()
    plot_results(history_data)