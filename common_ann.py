import numpy as np
import csv
import layers
import math
import imageio.v3 as iio
from pathlib import Path


def read_file(name, data=None):
    if data is None:
        with open(name) as csvfile:
            data = list(csv.reader(csvfile))
    data = np.asarray(data, dtype='int')
    np.random.seed(0)
    np.random.shuffle(data)
    n = len(data)
    X = data[:, 1:]
    # One-Hot Encoding
    Y = np.zeros((n, 10))
    for i in range(n):
        Y[i][data[i][0]] = 1
    return data, n, X, Y


def ADAM_init(sizeIn, sizeOut):
    fullyConnectedLayer = layers.FullyConnectedLayer(sizeIn, sizeOut)
    W = np.random.uniform(-math.sqrt(6 / (sizeIn + sizeOut)), math.sqrt(6 / (sizeIn + sizeOut)),
                          (sizeIn, sizeOut))
    b = np.zeros(sizeOut)
    fullyConnectedLayer.setWeights(W)
    fullyConnectedLayer.setBiases(b)
    sW = np.zeros((sizeIn, sizeOut))
    rW = np.zeros((sizeIn, sizeOut))
    sb = np.zeros(sizeOut)
    rb = np.zeros(sizeOut)
    return fullyConnectedLayer, sW, rW, sb, rb


def update_weight(W, s, r, epoch, dJdW):
    p1 = 0.9
    p2 = 0.999
    rate = 0.001
    delta = 1e-8
    s = p1 * s + (1 - p1) * dJdW
    r = p2 * r + (1 - p2) * dJdW ** 2
    W -= rate * (s / (1 - p1 ** epoch)) / (np.sqrt(r / (1 - p2 ** epoch)) + delta)


def ADAM_learning(X, dJdYhat, n, sW, rW, sb, rb, W, b, epoch):
    dJdW = np.matmul(np.transpose(X), dJdYhat) / n
    update_weight(W, sW, rW, epoch, dJdW)
    dJdb = np.sum(dJdYhat, axis=0) / n
    update_weight(b, sb, rb, epoch, dJdb)

def GTZAN_dataset():
    images = list()
    data_train = []
    data_test = []
    train_ratio = 70 # We know there are 100 files per class
    for class_idx,folder in enumerate(Path(r"GTZAN_Data\images_original_cropped").iterdir()):
        print(class_idx, folder) # Shows arbitrary classs number for the folder the dataset came from
        for img_idx,file in enumerate(Path(f"{folder}").iterdir()):
            if not file.is_file():
                continue
            image = iio.imread(file, mode='L')
            images.append(image)
            # Assigning class as first column to the flattened image list
            # Splitting the data into train and test
            if img_idx < train_ratio:
                data_train.append(np.array([class_idx]+list(image.flatten())))
            else:
                data_test.append(np.array([class_idx]+list(image.flatten())))
    return np.array(data_train), np.array(data_test)

def forward(X, Y, num_layer, customLayers, crossEntropy):
    Yhat = X
    for i in range(num_layer):
        Yhat = customLayers[i].forward(Yhat)
    crossEntropyLoss = crossEntropy.eval(Y, Yhat)
    return Yhat, crossEntropyLoss


def accuracy(n, X, Y, batch_size, num_layer, customLayers, crossEntropy):
    correct = 0
    for m_batch in range(10):
        begin_ = m_batch * batch_size
        end_ = (m_batch + 1) * batch_size
        Yhat, L = forward(X[begin_: end_], Y[begin_: end_], num_layer, customLayers, crossEntropy)
        Yhat_star = np.argmax(Yhat, axis=1)
        Y_star = np.argmax(Y[begin_: end_, :], axis=1)
        correct += sum([Yhat_star[i] == Y_star[i] for i in range(len(Yhat_star))])
    return correct / n * 100
