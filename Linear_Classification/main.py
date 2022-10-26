from pickletools import optimize
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt

def normalize(input:np.array):
    for i in range(0,len(input)):
        input[i] = (input[i].max() - input[i]) / (input[i].max() - input[i].min())

if __name__ == '__main__':
    data = load_breast_cancer()
    train_features, val_features, train_labels, val_labels = train_test_split(data['data'], data['target'], test_size=0.3)
    
    normalize(train_features)
    normalize(val_features)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(train_features.shape[1],)),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(train_features, train_labels,batch_size=16, epochs=100,validation_data=[val_features, val_labels])

    path = './log/'
    if not os.path.isdir(path):
        os.makedirs(path)

    model.save(os.path.join(path,'checkpoint.pth'))



    plt.plot(history.history['loss'], 'b', label='Train')
    plt.plot(history.history['val_loss'], 'r', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path,'loss_graph.jpg'))
    
    plt.clf()

    plt.plot(history.history['accuracy'], 'b', label='Train')
    plt.plot(history.history['val_accuracy'], 'r', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(path, 'acc_graph.jpg'))

    print('model trained with success!')

    

#N,D = x_train.shape

#model = tf.keras.models.Sequential([
#    tf.keras.layers.Input(shape=(D,)),
#   tf.keras.layers.Dense(1,activation='sigmoid')
#])

#model.compile(optimizer='adam',
#                loss='binary-crossentropy',
#                metrics=['accuracy'])
 