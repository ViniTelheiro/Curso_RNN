from re import T
import tensorflow as tf
from model import get_model
import json 
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def get_confusion_matrix(cm, classes, path:str, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    ticks_marks = np.arange(len(classes))
    plt.xticks(ticks_marks, classes, rotation=45)
    plt.yticks(ticks_marks, classes)
    
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], 'd'),
        horizontalalignment="center",
        color='white'if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)


if __name__ == '__main__':
    _, (test_features, test_labels) = tf.keras.datasets.mnist.load_data()

    model = get_model(checkpoint='./log/train/checkpoint.h5')

    _, acc = model.evaluate(test_features, test_labels,batch_size=32)

    output = model.predict(test_features, batch_size=32).argmax(axis=1)
    
    results = {'acc': acc}

    path = './log/test'
    if not os.path.isdir(path):
        os.makedirs(path)

    with open(os.path.join(path,'results.json'),'w') as json_file:
        json.dump(results, json_file, indent=2)
    json_file.close()
    
    cm = confusion_matrix(test_labels, output)
    get_confusion_matrix(cm, path=os.path.join(path, 'confusion_matrix.jpg'), classes=list(range(10)))

    print('model tested with success!')