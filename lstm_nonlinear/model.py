import tensorflow as tf

#input -> lstm -> dense, loss = mse lr = 0.05
def get_model():
    i = tf.keras.layers.Input(shape=(10, 1))
    x = tf.keras.layers.LSTM(10)(i)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(i, x)
    model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=0.05))
    return model
