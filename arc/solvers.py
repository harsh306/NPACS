import math
from scipy.optimize import fsolve, newton
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import *

T = 0.5
P = 0.1
v_0 = (8*T)/(3*P)# 0.31243171 #
rev = False
delta_l = 0.01

def arc_length_solver():
    return None

def np_solver(target_function,v_0, l):
    solutions = []
    for i in l:
        v = fsolve(target_function, v_0, i)
        #c = math.pow(v,2) + abs(i) = 1  
        solutions.append(v_0)
        v_0 = v
    return solutions

def newton_solver(target_function,v_0, l):
    """
    Default is secant method
    """
    solutions = []
    for i in l:
        v = newton(target_function,v_0, args=[i], tol=10**(-10), maxiter=10000)
        #c = math.pow(v,2) + abs(i) = 1  
        solutions.append(v_0)
        v_0 = v
    return solutions


def grad_solver(v_0, l, lr=0.01):
    solutions = []
    for i in l:
        v = v_0 + (lr * gas_h_jacobian(v_0,i))
        #c = math.pow(v,2) + abs(i) = 1  
        solutions.append(v_0)
        v_0 = v
    return solutions

def newton_raphson_solver(target_function,target_function_jacobian,v_0, l, lr=1):
    solutions = []
    for i in l:
        v = v_0 + lr*(target_function(v_0,i)/target_function_jacobian(v_0,i)) #newton(target_function,v_0, args=[i])
        #c = math.pow(v,2) + abs(i) = 1  
        solutions.append(v_0)
        v_0 = v
    return solutions

def secant_method_solver(f, x0, x1, iterations):
    xs = [x0, x1] + iterations * [None] # allocate x's vector

    for i, x in enumerate(xs):
        if i < iterations:
            x1 = xs[i+1]
            xs[i+2] = x1 - f(x1) * (x1 - x) / (f(x1) - f(x)) # fill component xs[i+2]
    
    return xs[-1] # return the now filled last entry
