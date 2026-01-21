import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
a=0
b=0
c=0

def calculateY(x,a,b,c):
    # f(x)=ax^2+bx+c
    return a*x**2+b*x+c

def generate_normal_data():
    x = np.linspace(-2, 2, 50)
    y_true=calculateY(x,a,b,c)
    noise=np.random.normal(0,1.5,size=len(x))
    y=y_true+noise
    return x,y,y_true


def generate_uniform_data():
    x = np.linspace(-2, 2, 50)
    y_true=calculateY(x,a,b,c)
    noise=np.random.uniform(-2.5,2.5,size=len(x))
    y=y_true+noise
    return x,y,y_true

def optimize(x,y,type):
    a_possible=np.linspace()
    b_possible=np.linspace()
    c_possible=np.linspace()

    best_loss=float('inf')

    for a in a_possible:
        for b in b_possible:
            for c in c_possible:

                y_guess=calculateY(x,a,b,c)
                loss=0
                e=np.abs(y-y_guess)

                if type=="L1":
                    loss=np.sum(e)
                elif type=="L2":
                    loss=np.sum(e**2)
                elif type=="L_inf":
                    loss=np.max(e)
                if loss < best_loss:
                    best_loss = loss
                    a_best,b_best,c_best = a, b, c
    return a_best,b_best,c_best

