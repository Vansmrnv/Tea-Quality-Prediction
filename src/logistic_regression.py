import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_and_gradient(X, y, w, b):
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

    return cost, dw, db

def gradient_descent(X, y, w, b, alpha, num_iterations):
    for i in range(num_iterations):
        cost, dw, db = compute_cost_and_gradient(X, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    return w, b

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









    
