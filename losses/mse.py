import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    # derivative of MSE w.r.t predictions
    return 2 * (y_pred - y_true) / y_true.size