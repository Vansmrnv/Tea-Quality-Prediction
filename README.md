# Tea-Quality-Prediction

# Tea Quality Prediction Using Logistic Regression and Neural Network

This project demonstrates the use of logistic regression to predict the quality of tea based on various features such as water temperature, steeping time, milk addition, and sugar addition. The goal is to classify whether a cup of tea is made correctly or not.

## Features

LR = Logistic Regression
NN = Neural Network

- **water_temperature**: Temperature of the water used (in degrees Celsius). LR/NN
- **steeping_time**: Time for which the tea is steeped (in minutes). LR/NN
- **milk**: Whether milk is added (1 for yes, 0 for no). NN
- **sugar**: Whether sugar is added (1 for yes, 0 for no). NN

## Model

The logistic regression model uses these features to predict whether the tea is made correctly (`1`) or not (`0`). The model is trained using gradient descent to optimize the weights and bias, and evaluated based on accuracy, precision, recall, and F1 score. The model is done to the most possible accuracy in the this situation. It could be seen on the sigmoid output graph that some values weren't identified correctly. This might be due to a non-linear relationship between the features, therefore it is better to use a neural network in this problem. 

## Repository Structure

- `data/`: Contains the dataset used for training and testing.
- `src/`: Contains the source code for the logistic regression model, prediction and data generation code.
- `images/`: Visualizations and plots related to the project.
- `README.md`: Overview and instructions for the project.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Jupyter Notebook
- NumPy
- Matplotlib
