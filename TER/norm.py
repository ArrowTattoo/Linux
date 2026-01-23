import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)
true_params=np.array([0,0,0])

def calculateY(x,params):
    a,b,c=params
    # f(x)=ax^2+bx+c
    return a*x**2+b*x+c

def generate_normal_data(true_params):
    x = np.linspace(-2, 2, 50)
    y_true=calculateY(x,true_params)
    noise=np.random.normal(0,1.5,size=len(x))
    y=y_true+noise
    return x,y,y_true


def generate_uniform_data(true_params):
    x = np.linspace(-2, 2, 50)
    y_true=calculateY(x,true_params)
    noise=np.random.uniform(-2.5,2.5,size=len(x))
    y=y_true+noise
    return x,y,y_true

def loss_l1(x,y,params):
    y_guess=calculateY(x,params)
    return np.sum(np.abs(y-y_guess))

def loss_l2(x,y,params):
    y_guess=calculateY(x,params)
    return np.sum((y-y_guess)**2)

def loss_l_inf(x,y,params):
    y_guess=calculateY(x,params)
    return np.max(np.abs(y-y_guess))

def experiment(data_type,):
    if data_type=="uniform":
        x,y,y_true=generate_uniform_data(true_params)
    else:
        x,y,y_true=generate_normal_data(true_params)
    initial = [1,1,1]
    res_l1 = minimize(loss_l1, initial, args=(x, y), method='Nelder-Mead')
    res_l2 = minimize(loss_l2, initial, args=(x, y), method='Nelder-Mead')
    res_l_inf = minimize(loss_l_inf, initial, args=(x, y), method='Nelder-Mead')

    e_l1 = np.sum(np.abs(true_params - res_l1.x))
    e_l2 = np.sum(np.abs(true_params - res_l2.x))
    e_l_inf = np.sum(np.abs(true_params - res_l_inf.x))





    
    # --- 绘图 ---
    # 画观测点
    ax.scatter(x, y_observed, color='gray', alpha=0.5, label='Observed Data (Points)')
    # 画真值线
    ax.plot(x, y_true, 'k--', linewidth=2, label='True Truth (Red Line Hidden)')
    
    # 画预测线
    ax.plot(x, model_func(x, res_l1.x), 'b-', alpha=0.8, label=f'L1 Fit (Err={err_l1:.2f})')
    ax.plot(x, model_func(x, res_l2.x), 'r-', alpha=0.8, label=f'L2 Fit (Err={err_l2:.2f})')
    ax.plot(x, model_func(x, res_linf.x), 'g-', alpha=0.8, label=f'L_inf Fit (Err={err_linf:.2f})')
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return {
        "L1_Error": err_l1,
        "L2_Error": err_l2,
        "L_inf_Error": err_linf
    }

# 4. 主程序
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 实验 1: 正态分布
res_normal = run_experiment('Normal', axes[0])

# 实验 2: 均匀分布
res_uniform = run_experiment('Uniform', axes[1])

plt.tight_layout()
print("=== 实验结果: 参数误差 (越小越好) ===")
print(f"正态分布环境: \n  L1误差: {res_normal['L1_Error']:.4f}\n  L2误差: {res_normal['L2_Error']:.4f}\n  L_inf误差: {res_normal['L_inf_Error']:.4f}")
print("-" * 30)
print(f"均匀分布环境: \n  L1误差: {res_uniform['L1_Error']:.4f}\n  L2误差: {res_uniform['L2_Error']:.4f}\n  L_inf误差: {res_uniform['L_inf_Error']:.4f}")

plt.show() # 显示图像