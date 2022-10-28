import tensorflow as tf
import numpy as np

def get_model():
    i = tf.keras.layers.Input(shape=(10, ))
    x = tf.keras.layers.Dense(1)(i)
    model = tf.keras.Model(i, x)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    return model

def get_ann_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(128,activation='relu'),
      tf.keras.layers.Dropout(.2),
      tf.keras.layers.Dense(1)  
    ])
    
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model