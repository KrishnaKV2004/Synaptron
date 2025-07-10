import numpy as np

def linear(x):
    return x

def linear_derivative(x):
    
    if hasattr(x, "__len__"):
        return np.ones_like(x)
    else:
        return 1