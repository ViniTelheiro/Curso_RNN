import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from model import get_models, get_model_2
import json
from sklearn.metrics import confusion_matrix
import itertools

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint.h5 path')
    return parser.parse_args()


def get_confusion_matrix(cm, classes, path:str, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    ticks_mark = np.arange(len(classes))
    plt.xticks(ticks_mark, classes, rotation=45)
    plt.yticks(ticks_mark, classes)

    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], 'd'),
        horizontalalignment='center',
        color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)


if __name__ == '__main__':
    args = get_args()
    
    if 'model_1'in args.checkpoint:
        model = get_models()
        path = './log/test/model_1'
    else:
        model = get_model_2()
        path = './log/test/model_2'
    
    if not os.path.isdir(path):
        os.makedirs(path)
    
    model.load_weights(args.checkpoint)

    dataset = tf.keras.datasets.mnist.load_data()
    features = dataset[1][0]
    labels = dataset[1][1]

    predict = model.predict(features,batch_size=32).argmax(axis=1)

    metrics = model.evaluate(features,labels, batch_size=32)

    results = {
        'chekpoint': args.checkpoint,
        'acc':metrics[1]
    }

    with open(os.path.join(path,'results.json'), 'w') as json_file:
        json.dump(results, json_file, indent=2)
    json_file.close()

    print(results)

    cm = confusion_matrix(labels, predict)
    get_confusion_matrix(cm, classes=list(range(10)), path=os.path.join(path,'confusion_matrix.jpg'))

    print('model tested with success!')    
    
