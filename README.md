# Tea-Quality-Prediction

# Tea Quality Prediction Using Logistic Regression

This project demonstrates the use of logistic regression to predict the quality of tea based on various features such as water temperature, steeping time, milk addition, and sugar addition. The goal is to classify whether a cup of tea is made correctly or not.

## Features

- **water_temperature**: Temperature of the water used (in degrees Celsius).
- **steeping_time**: Time for which the tea is steeped (in minutes).
- **milk**: Whether milk is added (1 for yes, 0 for no).
- **sugar**: Whether sugar is added (1 for yes, 0 for no).

## Model

The logistic regression model uses these features to predict whether the tea is made correctly (`1`) or not (`0`). The model is trained using gradient descent to optimize the weights and bias, and evaluated based on accuracy, precision, recall, and F1 score.

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
