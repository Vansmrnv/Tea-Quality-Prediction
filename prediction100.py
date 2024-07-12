import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from src.logistic_regression import sigmoid, compute_cost_and_gradient, gradient_descent, predict, evaluate_model

## Step 2: Load Data
file_path = 'data/tea_data.csv'
data = pd.read_csv(file_path)
print(data.describe())
X = data[['water_temperature', 'steeping_time']].values
y = data['tea_made_correctly'].values

# Generate Polynomial Features for non-linear boudary using sklearn
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)


# Feature scaling (standardization)


# Same, except we scale x_poly using in built library
scaler = StandardScaler()
X_poly = scaler.fit_transform(X_poly)

## Initialize Parameters
w = np.zeros(X_poly.shape[1])
b = 0
alpha = 1
num_iterations = 1000
lambda_ = 0 #Regularized to make the boundary fit better (spoiler: it is overfitting a bit) in order not to overfit lambda_ has to be 0

# Train Model
w, b, cost_history = gradient_descent(X_poly, y, w, b, alpha, num_iterations, lambda_)

plt.plot(range(num_iterations), cost_history, label='Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.legend()
plt.show()

# Evaluate Model
accuracy, precision, recall, f1_score, TP, FP, TN, FN = evaluate_model(X_poly, y, w, b)

# Print Results
print(f"Accuracy: {accuracy}%")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Confusion Matrix:\nTP: {TP}, FP: {FP}\nFN: {FN}, TN: {TN}")

# Visualize Results
# Create a mesh grid for plotting decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict using the logistic regression model with polynomial features
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_poly = poly.transform(grid_points)
grid_points_poly = scaler.transform(grid_points_poly)
Z = predict(grid_points_poly, w, b)
Z = Z.reshape(xx.shape)

# Plot decision boundary and data points
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color='g', label='Correctly Made Tea')
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color='r', label='Incorrectly Made Tea')
plt.xlabel('Standardized Water Temperature')
plt.ylabel('Standardized Steeping Time')
plt.legend()
plt.title('Logistic Regression Decision Boundary with Polynomial Features')
plt.show()

z = np.linspace(-10, 10, 100)
sigmoid_z = sigmoid(z)
plt.plot(z, sigmoid_z, label='Sigmoid Function')

# Add bias term to X and transform with polynomial features
X_b = poly.transform(X)  # Generate polynomial features
X_b = scaler.transform(X_b)  # Standardize polynomial features
z_data = np.dot(X_b, w) + b
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
