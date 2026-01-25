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

def loss_l1(params,x,y):
    y_guess=calculateY(x,params)
    return np.sum(np.abs(y-y_guess))

def loss_l2(params,x,y):
    y_guess=calculateY(x,params)
    return np.sum((y-y_guess)**2)

def loss_l_inf(params,x,y):
    y_guess=calculateY(x,params)
    return np.max(np.abs(y-y_guess))

def experiment(data_type,ax):
    if data_type=="Uniform":
        x,y,y_true=generate_uniform_data(true_params)
        title = "Uniform"
    else:
        x,y,y_true=generate_normal_data(true_params)
        title = "Normal"
    initial = [1,1,1]
    res_l1 = minimize(loss_l1, initial, args=(x, y), method='Nelder-Mead')
    res_l2 = minimize(loss_l2, initial, args=(x, y), method='Nelder-Mead')
    res_l_inf = minimize(loss_l_inf, initial, args=(x, y), method='Nelder-Mead')

    e_l1 = np.sum(np.abs(true_params - res_l1.x))
    e_l2 = np.sum(np.abs(true_params - res_l2.x))
    e_l_inf = np.sum(np.abs(true_params - res_l_inf.x))
    ax.set_title(title)

    ax.scatter(x, y, color='gray', alpha=0.5, label='Points')
    ax.plot(x, y_true, 'k--', linewidth=2, label='True Truth ')
    
    ax.plot(x, calculateY(x, res_l1.x), 'b-', alpha=0.8, label=f'L1 (E={e_l1:.2f})')
    ax.plot(x, calculateY(x, res_l2.x), 'r-', alpha=0.8, label=f'L2 (E={e_l2:.2f})')
    ax.plot(x, calculateY(x, res_l_inf.x), 'g-', alpha=0.8, label=f'L_inf (E={e_l_inf:.2f})')


    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return {
        "L1_Error": e_l1,
        "L2_Error": e_l2,
        "L_inf_Error": e_l_inf
    }


plt.figure(figsize=(8, 6)) 
ax1 = plt.gca() 
res_normal = experiment('Normal', ax1) 


plt.figure(figsize=(8, 6))
ax2 = plt.gca() 
res_uniform = experiment('Uniform', ax2) 

print(f"Normal e_L1: {res_normal['L1_Error']:.4f},e_L2: {res_normal['L2_Error']:.4f},e_L_inf: {res_normal['L_inf_Error']:.4f}")
print(f"Uniform e_L1: {res_uniform['L1_Error']:.4f},e_L2: {res_uniform['L2_Error']:.4f},e_L_inf: {res_uniform['L_inf_Error']:.4f}")

plt.show()