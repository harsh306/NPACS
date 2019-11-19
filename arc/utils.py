import math
from scipy.optimize import fsolve, newton
import numpy as np
import random
import matplotlib.pyplot as plt



T = 0.5
P = 0.1
v_0 = (8*T)/(3*P)# 0.31243171 #
rev = False
delta_l = 0.01


def get_l(minor,major,delta_l, rev=False):
    """
    uniformly seperated values for lambda
    """
    l = np.arange(minor, major+delta_l, delta_l)
    if rev:
        return l[ : :-1]
    else:
        return l

def vdwg_f(v):
    """
    Van der Waals Gas
    """
    return (P + (3/(v*v))) * (v - (1/3))


def vdwg_f_jacobian(v):
    """
    Van der Waals Gas jacobian
    """
    try:
        vdwg_f_jacobian = P - 3* (1/(math.pow(v,-2))) + 2* (math.pow(v,-3))
    except:
        vdwg_f_jacobian = 1e-10
    return vdwg_f_jacobian

def ig_g_jacobian(v):
    """
    Ideal Gas jacobian
    """
    try:
        ig_g_jacobian = P - (8 * T)/(3 * P)
    except:
        ig_g_jacobian = 1e-10
    return ig_g_jacobian

def gas_h_jacobian(v,l):
    """
    """
    return l*vdwg_f_jacobian(v) + (1-l)* ig_g_jacobian(v)

def detect_bifurcation_static(solutions):
    """
    Detect singularities in homotopy path. 
    """
    jacobians = []
    for v in solutions:
        #compute and store Jacobian
        jacobians.append(round(vdwg_f_jacobian(v)))
    j_sign = np.sign(jacobians)
    signchange = ((np.roll(j_sign, 1) - j_sign) != 0).astype(int)
    return np.where(signchange==1), j_sign, jacobians

def ig_g(v):
    """
    Ideal Gas
    """
    return (P*v) - ((8/3) * T)

def gas_h(v,l):
    """
    Ideal gas to Van der Waals Gas
    """
    return l*vdwg_f(v) + (1-l)*ig_g(v)

def cubic_h(v,l):
    """
    cubic polynomial 
    """
    return math.pow(v,3) - (l * v)

def plot_solutions(l,solutions):
    return plt.scatter(l,solutions)


    