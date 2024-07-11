import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

model = Sequential([
  Dense(units=8, activation='relu'),
  Dense(units=8, activation='relu'),
  Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate='1e-3'),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

model.fit(X, Y, epochs=50)
