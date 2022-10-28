import numpy as np
from synthetic_dataset import synthetic_dataset_linear as synthetic_dataset
from model import get_linear_model
import matplotlib.pyplot as plt
import tensorflow as tf
import os


if __name__ == "__main__":
    series, X, Y = synthetic_dataset(noise=True)

    N = len(X)

    model = get_linear_model()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        X[:-N//2], Y[:-N//2],
        epochs=100,
        validation_data=(X[-N//2:], Y[-N//2:]),
        callbacks=[callback]
    )

    path = './log/Linear_module'
    
    if not os.path.isdir(path):
        os.makedirs(path)

    model.save(os.path.join(path,'checkpoint.h5'))

    plt.plot(history.history['loss'], 'b', label='Train')
    plt.plot(history.history['val_loss'], 'r', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss_graph.jpg'))
    plt.clf()
     
    print('Model trained with success!')

    #forecasting
    validation_target = Y[-N//2:]
    validation_predictions = []

    last_x = X[-N//2]
    while len(validation_predictions) < len(validation_target):
        p = model.predict(last_x.reshape(1,-1))[0,0]
        validation_predictions.append(p)
        last_x = np.roll(last_x,-1)
        last_x[-1] = p

    plt.plot(validation_target, 'b', label='Target')
    plt.plot(validation_predictions, 'r', label='Prediction')
    plt.legend()
    plt.savefig(os.path.join(path, 'predictions.jpg'))

    print('prediction done with success!')