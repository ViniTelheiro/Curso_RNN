import tensorflow as tf
import os
import matplotlib.pyplot as plt
from model import get_model


if __name__ == '__main__':

    (train_features, train_labels), _ = tf.keras.datasets.mnist.load_data()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model = get_model()
    
    history = model.fit(train_features, train_labels, batch_size=32, epochs=100,validation_split=.10, callbacks=[callback])

    path='./log/train'
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

    plt.plot(history.history['accuracy'], 'b', label='Train')
    plt.plot(history.history['val_accuracy'], 'r', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(path, 'acc_graph.jpg'))

    print('Model trained with success!')
