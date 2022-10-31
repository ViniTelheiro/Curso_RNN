import tensorflow as tf
import numpy as np
from dataset import dataset
from model import get_model
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    series, X, Y = dataset()

    N = len(X)

    model = get_model()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    r = model.fit(
        X[:-N//2], Y[:-N//2],
        batch_size = 32,
        epochs = 100,
        validation_data=(X[-N//2:], Y[-N//2:]),
        callbacks=[callback]
    )

    path = './log/'
    if not os.path.isdir(path):
        os.makedirs(path)
    
    model.save(os.path.join(path, 'checkpoint.h5'))

    plt.plot(r.history['loss'], 'b', label='Train')
    plt.plot(r.history['val_loss'], 'r', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss_graph.jpg'))
    plt.clf()

    print('model trained with success!')
    print('forecasting values...')

    #one step
    outputs = model.predict(X)
    predictions = outputs[:,0]

    
    plt.plot(Y, 'b', label='Target')
    plt.plot(predictions, 'r', label='Predictions')
    plt.legend()
    plt.savefig(os.path.join(path, 'one_step_forecastiong.jpg'))
    plt.clf()

    #multi_step
    validation_predictions = []
    last_x = X[-N//2]
    validation_target = Y[-N//2:]
    
    while len(validation_predictions) < len(validation_target):
        p = model.predict(last_x.reshape(1,-1,1))[0,0]
        validation_predictions.append(p)
        last_x = np.roll(last_x, -1)
        last_x[-1] = p

    plt.plot(validation_target, 'b', label='Target')
    plt.plot(validation_predictions, 'r', label='Predictions')
    plt.legend()
    plt.savefig(os.path.join(path, 'multi_step_forecastiong.jpg'))
    plt.clf()

