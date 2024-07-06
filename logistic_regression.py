import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(z):
  g = 1/(1+np.exp(-z))
  return g

def compute_cost_and_gradient(X, y, w, b):
    m = X.shape[0]
    cost = 0
    dw = np.zeros(w.shape)
    db = 0
    for i in range(m):
        z_i = np.dot(X[i], w)+b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)

        dw += (f_wb_i - y[i]) * X[i].reshape(-1, 1)
        db += (f_wb_i - y[i])


    cost = cost/m

    dw = dw/m
    db = db/m

    return cost, dw , db
    
