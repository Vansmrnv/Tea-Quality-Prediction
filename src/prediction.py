import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.logistic_regression import sigmoid, compute_cost_and_gradient, gradient_descent, predict, evaluate_model

## Step 2: Load Data
file_path = 'data/tea_data.csv'
data = pd.read_csv(file_path)
print(data.describe())
X = data[['water_temperature', 'steeping_time']].values
y = data['tea_made_correctly'].values

# Feature scaling (standardization)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

## Step 3: Initialize Parameters
w = np.zeros(X.shape[1])
b = 0
alpha = 0.1
num_iterations = 1000

## Step 4: Train Model
w, b, cost_history = gradient_descent(X, y, w, b, alpha, num_iterations)

plt.plot(range(num_iterations), cost_history, label='Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.legend()
plt.show()

## Step 5: Evaluate Model
accuracy, precision, recall, f1_score, TP, FP, TN, FN = evaluate_model(X, y, w, b)

## Step 6: Print Results
print(f"Accuracy: {accuracy}%")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Confusion Matrix:\nTP: {TP}, FP: {FP}\nFN: {FN}, TN: {TN}")

## Step 7: Visualize Results
# Sigmoid Function Plot
z = np.linspace(-10, 10, 100)
sigmoid_z = sigmoid(z)
plt.plot(z, sigmoid_z, label='Sigmoid Function')

# Data Points
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
z_data = np.dot(X_b, np.r_[b, w])
sigmoid_data = sigmoid(z_data)

# Separate positive and negative examples
pos = y == 1
neg = y == 0

plt.scatter(z_data[pos], sigmoid_data[pos], c='g', label='Positive (Correct Tea)', alpha=0.5)
plt.scatter(z_data[neg], sigmoid_data[neg], c='r', label='Negative (Incorrect Tea)', alpha=0.5)

plt.xlabel('z')
plt.ylabel('sigma(z)')
plt.title('Sigmoid Function with Data Points')
plt.legend()
plt.show()
