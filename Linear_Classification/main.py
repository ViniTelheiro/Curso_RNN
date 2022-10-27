from pickletools import optimize
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json

def normalize(input:np.array):
    for i in range(0,len(input)):
        input[i] = (input[i].max() - input[i]) / (input[i].max() - input[i].min())


def get_errors(predicted:np.array, labels:np.array, errors:dict):
    for i in range(0,len(predicted)):
        p = int(predicted[i])
        l = int(labels[i])
        if p == l:
            if p == 0:
                errors['tn'] += 1
            else:
                errors['tp'] += 1
        else:
            if p == 0:
                errors['fn'] += 1
            else:
                errors['fp'] += 1


if __name__ == '__main__':
    data = load_breast_cancer()
    train_features, test_features, train_labels, test_labels = train_test_split(data['data'], data['target'], test_size=0.3)
    
    normalize(train_features)
    normalize(test_features)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(train_features.shape[1],)),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(train_features, train_labels,batch_size=32, epochs=100,validation_split=0.1,callbacks=[callback])

    path = './log/train'
    if not os.path.isdir(path):
        os.makedirs(path)

    model.save(os.path.join(path,'checkpoint.h5'))

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


    _, acc = model.evaluate(test_features, test_labels, batch_size=32)
    
    output = model.predict(test_features)
    _, predicted = cv2.threshold(output, 0.5, 1, cv2.THRESH_BINARY)
    predicted = predicted.squeeze(-1)

    errors = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
    
    get_errors(predicted, test_labels, errors)
    pos = int(np.sum(test_labels))
    neg = int(len(test_labels) - pos)

    results = {
        'acc': float(acc),
        'pos': pos,
        'neg': neg,
        'tp': errors['tp'],
        'tn': errors['tn'],
        'fp': errors['fp'],
        'fn': errors['fn']
    }
    path = './log/test'
    
    if not os.path.isdir(path):
        os.makedirs(path)

    with open(os.path.join(path, 'results.json'), 'w') as json_file:
        json.dump(results, json_file, indent=2)
    json_file.close()

    print('model tested with success!')