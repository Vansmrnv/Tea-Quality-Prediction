import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer   
import matplotlib.pyplot as plt
#from tensorflow.keras.utils import build_models

# Load Data
data = pd.read_csv('data/tea_data500.csv')

# Define features and target
X = data[['water_temperature', 'steeping_time']]
y = data['tea_made_correctly']

y = np.expand_dims(y, axis=1)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

del X_temp, y_temp

print(f"the shape of the training set (input) is: {X_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {X_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {X_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_cv_scaled = scaler.fit_transform(X_cv)
X_test_scaled = scaler.fit_transform(X_test)

# Build the neural network model

nn_train_error = []
nn_cv_error = []

model = Sequential([
    Dense(units=25, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(units=15, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(units=1, activation='sigmoid', kernel_regularizer=L2(0.01))
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
print(f'Training {model.name}...')

model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

print('Done!\n')

threshold = 0.5

yhat = model.predict(X_train_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= threshold, 1, 0)
train_error = np.mean(yhat != y_train)
nn_train_error.append(train_error)

yhat = model.predict(X_cv_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= threshold, 1,0)
cv_error = np.mean(yhat != y_cv)
nn_cv_error.append(cv_error)

for model_nem in range(len(nn_train_error)):
    print(
        f'Model {model_nem+1}: Training set Classification Error: {nn_train_error[model_nem]:.5f}, ' +
        f'CV Set Classification Error: {nn_cv_error[model_nem]:.5f}'
    )


def eval_cat_err(y, yhat):   # Evaluates categorical error 
    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i] != y[i]:
            incorrect += 1
        cerr = incorrect/m
    return cerr

model = Sequential(
    [
        Dense(units=120, activation='relu', name='L1'),
        Dense(units=40, activation='relu', name='L2'),
        Dense(units=6, activation='linear', name='L3')
    ],name="Complex"
)

model.compile( 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01))

model.fit(X_train, y_train, epochs=100)
