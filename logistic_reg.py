import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_and_gradient(X, y, w, b, lambda_):
    m = X.shape[0]
    cost = 0
    dw = np.zeros(w.shape)
    db = 0

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
        dw += (f_wb_i - y[i]) * X[i]
        db += (f_wb_i - y[i])
    
    cost = cost / m
    dw = dw / m
    db = db / m

    cost += (lambda_ / (2*m))*np.sum(w**2)
    dw += (lambda_ /m)*w 

    return cost, dw, db

def gradient_descent(X, y, w_in, b_in, alpha, num_iterations, lambda_):
    m = len(y)
    w = copy.deepcopy(w_in)
    b = b_in
    cost_history = []
    for i in range(num_iterations):
        cost, dw, db = compute_cost_and_gradient(X, y, w, b, lambda_)
        w -= alpha * dw
        b -= alpha * db
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
        cost_history.append(cost)
    return w, b, cost_history

def predict(X, w, b):
    m = X.shape[0]
    prediction = np.zeros(m)
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        prediction[i] = 1 if f_wb_i > 0.5 else 0
    return prediction

def evaluate_model(X_test, y_test, w, b):
    prediction = predict(X_test, w, b)
    accuracy = np.mean(prediction == y_test) * 100
    TP = np.sum((y_test == 1) & (prediction == 1))
    FP = np.sum((y_test == 0) & (prediction == 1))
    TN = np.sum((y_test == 0) & (prediction == 0))
    FN = np.sum((y_test == 1) & (prediction == 0))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score, TP, FP, TN, FN
