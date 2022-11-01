import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
from model import get_model_2 as get_models
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    dataset = tf.keras.datasets.mnist.load_data()
    
    features = np.array(dataset[0][0])
    labels = np.array(dataset[0][1])

    model = get_models()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    r = model.fit(features,labels, batch_size=32, epochs=100, callbacks=[callback], validation_split=.1)

    path = './log/train/model_2'
    if not os.path.isdir(path):
        os.makedirs(path)
    
    model.save(os.path.join(path, 'checkpoint.h5'))

    plt.plot(r.history['loss'], 'b', label='Train')
    plt.plot(r.history['val_loss'], 'r', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path,'loss_graph.jpg'))
    plt.clf()
    
    plt.plot(r.history['acc'], 'b', label='Train')
    plt.plot(r.history['val_acc'], 'r', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(path,'acc_graph.jpg'))
    plt.clf()

    print('model trained with success!')