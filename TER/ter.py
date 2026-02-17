import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)       
TRUE_PARAMS = np.array([1, 0, 2]) # a=1, b=0, c=2
NB_DATA = 40             #
NB_EXPERIMENTS = 100     

def calculateY(x, params):
    a, b, c = params
    return a * x**2 + b * x + c


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

def generate_laplace_data(true_params,NB_Data):
    x = np.linspace(-2, 2, NB_Data)
    y_true = calculateY(x, true_params)
    noise = np.random.laplace(0, 0.5, size=len(x))
    y = y_true + noise
    return x, y, y_true

def loss_l2(params,x,y):
    y_guess=calculateY(x,params)
    return np.sum((y-y_guess)**2)

def solve_l2(x, y):
    initial = [1,1,1]
    res = minimize(loss_l2, initial, args=(x,y), method='BFGS')
    return res.x

def objective_linf(p): return p[3]

def constraint_linf(x,y,p):
        err = y - calculateY(x, p[:3])
        t = p[3]
        return np.concatenate([t - err, t + err])

def solve_linf(x, y):
    # min t
    # -t <= error <= t
    initial = [1, 1, 1, 5] 
    cons = {'type': 'ineq', 'fun': lambda p: constraint_linf(x, y, p)}
    res = minimize(objective_linf, initial, method='SLSQP', constraints=cons)
    return res.x[:3]




def objective_l1(p): 
        return np.sum(p[3:]) 
    
def constraint_l1(x,y,p):
    a, b, c = p[:3]
    t_i = p[3:]
    err = y - (a * x**2 + b * x + c)
    return np.concatenate([t_i - err, t_i + err])

def solve_l1(x, y):
    # min sum(t_i)
    # -t_i <= error_i <= t_i
    n = len(x)
    initial = np.concatenate([[1, 1, 1], np.ones(n)])
    
    cons = {'type': 'ineq', 'fun': lambda p: constraint_l1(x, y, p)}
    res = minimize(objective_l1, initial, method='SLSQP', constraints=cons, options={'maxiter': 1000})
    return res.x[:3]

def experiment(data_type, ax, NB_Data):
    if data_type == "Uniform":
        x, y, y_true = generate_uniform_data(TRUE_PARAMS, NB_Data)
        title = "Uniform Distribution (Best: L_inf)"
    elif data_type == "Laplace":
        x, y, y_true = generate_laplace_data(TRUE_PARAMS, NB_Data)
        title = "Laplace Distribution (Best: L1)"
    else:
        x, y, y_true = generate_normal_data(TRUE_PARAMS, NB_Data)
        title = "Normal Distribution (Best: L2)"
    
    params_l1 = solve_l1(x, y)
    params_l2 = solve_l2(x, y)
    params_linf = solve_linf(x, y)

    e_l1 = np.linalg.norm(TRUE_PARAMS - params_l1)
    e_l2 = np.linalg.norm(TRUE_PARAMS - params_l2)
    e_l_inf = np.linalg.norm(TRUE_PARAMS - params_linf)

    ax.set_title(title + "\nnb_data=" + str(NB_Data))
    ax.scatter(x, y, color='gray', alpha=0.5, label='Points')
    ax.plot(x, y_true, 'k--', linewidth=2, label='True Truth')
    
    ax.plot(x, calculateY(x, params_l1), 'b-', alpha=0.8, linewidth=1.5, label=f'L1 (E={e_l1:.2f})')
    ax.plot(x, calculateY(x, params_l2), 'r-', alpha=0.8, linewidth=1.5, label=f'L2 (E={e_l2:.2f})')
    ax.plot(x, calculateY(x, params_linf), 'g-', alpha=0.8, linewidth=1.5, label=f'L_inf (E={e_l_inf:.2f})')

    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return {
        "L1_Error": e_l1,
        "L2_Error": e_l2,
        "L_inf_Error": e_l_inf
    }


def run_simulation():
    distributions = ['Normal', 'Uniform', 'Laplace']
    solvers = ['L1', 'L2', 'L_inf']
    
    history = {dist: {solver: [] for solver in solvers} for dist in distributions}
    
    for i in range(NB_EXPERIMENTS):
        for dist_type in distributions:
            if dist_type == 'Normal':
                x, y, _ = generate_normal_data(TRUE_PARAMS, NB_DATA)
            elif dist_type == 'Uniform':
                x, y, _ = generate_uniform_data(TRUE_PARAMS, NB_DATA) 
            else:
                x, y, _ = generate_laplace_data(TRUE_PARAMS, NB_DATA)

            p_l1 = solve_l1(x, y)
            err_l1 = np.linalg.norm(TRUE_PARAMS - p_l1)
            
            p_l2 = solve_l2(x, y)
            err_l2 = np.linalg.norm(TRUE_PARAMS - p_l2)
        
            p_linf = solve_linf(x, y)
            err_linf = np.linalg.norm(TRUE_PARAMS - p_linf)
            
            history[dist_type]['L1'].append(err_l1)
            history[dist_type]['L2'].append(err_l2)
            history[dist_type]['L_inf'].append(err_linf)
            
    return history


def plot_results(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dists = ['Normal', 'Uniform', 'Laplace']
    solvers = ['L1', 'L2', 'L_inf']
    
    print("\n" + "="*45)
    print(f"{'Dist':<10} | {'L1':<8} | {'L2':<8} | {'L_inf':<8}")
    print("-" * 45)

    for ax, dist in zip(axes, dists):
        means = [np.mean(history[dist][s]) for s in solvers]
        stds = [np.std(history[dist][s]) for s in solvers]
        
        bars = ax.bar(solvers, means, yerr=stds, capsize=5, color=['blue','red','green'], alpha=0.6)
        
        best_idx = np.argmin(means)
        ax.set_title(f"{dist}\nWinner: {solvers[best_idx]}", weight='bold')
        ax.bar_label(bars, fmt='%.2f')
        
        marks = [f"*{m:.3f}*" if i==best_idx else f"{m:.3f}" for i, m in enumerate(means)]
        print(f"{dist:<10} | {marks[0]:<8} | {marks[1]:<8} | {marks[2]:<8}")
    print("="*45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running single experiment visualization...")
    plt.figure(figsize=(8, 6)) 
    ax1 = plt.gca() 
    res_normal = experiment('Normal', ax1, NB_DATA) 
    plt.show()

    plt.figure(figsize=(8, 6))
    ax2 = plt.gca() 
    res_uniform = experiment('Uniform', ax2, NB_DATA) 
    plt.show()

    plt.figure(figsize=(8, 6))
    ax3 = plt.gca()
    res_laplace = experiment('Laplace', ax3, NB_DATA)
    plt.show()

    print("-" * 60)
    print(f"Normal  Result -> L1: {res_normal['L1_Error']:.4f}, L2: {res_normal['L2_Error']:.4f}, L_inf: {res_normal['L_inf_Error']:.4f}")
    print(f"Uniform Result -> L1: {res_uniform['L1_Error']:.4f}, L2: {res_uniform['L2_Error']:.4f}, L_inf: {res_uniform['L_inf_Error']:.4f}")
    print(f"Laplace Result -> L1: {res_laplace['L1_Error']:.4f}, L2: {res_laplace['L2_Error']:.4f}, L_inf: {res_laplace['L_inf_Error']:.4f}")
    print("-" * 60)

    # Part 2: Run Monte Carlo Simulation 
    history_data = run_simulation()
    plot_results(history_data)