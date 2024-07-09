import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.logistic_regression import sigmoid, compute_cost_and_gradient, gradient_descent, predict, evaluate_model

file_path = 'tea_data.csv'
data = pd.read_csv(file_path)
X = data[['water_temperature', 'steeping_time']].values
y = data['tea_made_correctly'].values

w = np.zeros(X.shape[1])
b = 0
alpha = 0.01
num_iterations = 1000

w, b = gradient_descent(X, y, w, b, alpha, num_iterations)

accuracy, precision, recall, f1_score, TP, FP, TN, FN = evaluate_model(X, y, w, b)

print(f"Accuracy: {accuracy}%")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Confusion Matrix:\nTP: {TP}, FP: {FP}\nFN: {FN}, TN: {TN}")

z = np.linspace(-10, 10, 100)
sigmoid_z = sigmoid(z)
plt.plot(z, sigmoid_z, label='Sigmoid Function')

X_b = np.c_[np.ones((X.shape[0], 1)), X]  
z_data = np.dot(X_b, np.r_[b, w])
sigmoid_data = sigmoid(z_data)

pos = y == 1
neg = y == 0

plt.scatter(z_data[pos], sigmoid_data[pos], c='g', label='Positive (Correct Tea)')
plt.scatter(z_data[neg], sigmoid_data[neg], c='r', label='Negative (Incorrect Tea)')

plt.xlabel('z')
plt.ylabel('sigma(z)')
plt.title('Sigmoid Function with Data Points')
plt.legend()
plt.show()
