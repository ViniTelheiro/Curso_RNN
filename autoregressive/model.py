import tensorflow as tf
import numpy as np

def get_linear_model():
    i = tf.keras.layers.Input(shape=(10, ))
    x = tf.keras.layers.Dense(1)(i)
    model = tf.keras.Model(i, x)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    
    return model

def get_rnn_model():
    i = tf.keras.layers.Input(shape=(10, 1))
    x = tf.keras.layers.SimpleRNN(5, activation='relu')(i)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(i, x)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    return model
    