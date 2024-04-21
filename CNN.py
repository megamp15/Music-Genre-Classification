import os
import cv2
import numpy as np
import layers
import common
import matplotlib.pyplot as plt
from PIL import Image
import time

def create_dataset():
    X_train = np.zeros((700, 218, 336))
    X_valid = np.zeros((300, 218, 336))
    Y_train = np.zeros((700, 10))
    Y_valid = np.zeros((300, 10))

    img_folder = "GTZAN_Data/images_original_cropped"
    cur_y = 0
    n_train = 0
    n_valid = 0
    for dirr in os.listdir(img_folder):
        cnt_dir = 0
        for file in os.listdir(os.path.join(img_folder, dirr)):
            img_path = os.path.join(img_folder, dirr, file)
            img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
            # normalize the pixel
            img /= 255
            if cnt_dir < 70:
                X_train[n_train] = img
                Y_train[n_train][cur_y] = 1
                n_train += 1
            else:
                X_valid[n_valid] = img
                Y_valid[n_valid][cur_y] = 1
                n_valid += 1
            cnt_dir += 1
        if cnt_dir == 99:
            X_valid[n_valid] = img
            Y_valid[n_valid][cur_y] = 1
            n_valid += 1
        cur_y += 1
    #shuffle data
    np.random.seed(0)
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    return n_train, X_train, Y_train, n_valid, X_valid, Y_valid


if __name__ == '__main__':
    n_train, X_train, Y_train, n_valid, X_valid, Y_valid = create_dataset()
    customLayers = [None] * 10
    convolutionalLayer = layers.ConvolutionalLayer()
    customLayers[0] = convolutionalLayer
    customLayers[1] = layers.ReLuLayer()
    customLayers[2] = layers.MaxPoolLayer()
    customLayers[3] = layers.ConvolutionalLayer()
    customLayers[4] = layers.ReLuLayer()
    customLayers[5] = layers.MaxPoolLayer()
    customLayers[6] = layers.FlattenedLayer()
    customLayers[7] = layers.DropoutLayer()
    fullyConnectedLayer, sW, rW, sb, rb = common.ADAM_init(828, 10)
    customLayers[8] = fullyConnectedLayer
    customLayers[9] = layers.SoftmaxLayer()
    crossEntropy = layers.CrossEntropy()
    L_train = np.zeros(101)
    L_valid = np.zeros(101)
    for epoch in range(1, 10):
        print("epoch:", epoch)
        for m_batch in range(10):
            begin_valid = m_batch * 30
            end_valid = (m_batch + 1) * 30
            Yhat_valid, l_valid = common.forward(X_valid[begin_valid:end_valid], Y_valid[begin_valid:end_valid],
                                                 len(customLayers), customLayers, crossEntropy)
            L_valid[epoch] += l_valid
            begin_train = m_batch * 70
            end_train = (m_batch + 1) * 70
            Yhat_train, l_train = common.forward(X_train[begin_train:end_train], Y_train[begin_train:end_train],
                                                 len(customLayers), customLayers, crossEntropy)
            L_train[epoch] += l_train
            # backward
            dJdYhat = crossEntropy.gradient(Y_train[begin_train:end_train], Yhat_train)
            for i in reversed(range(len(customLayers))):
                if customLayers[i] == fullyConnectedLayer:
                    # fullyConnectedLayer.updateWeights(dJdYhat, 0.001)
                    common.ADAM_learning(fullyConnectedLayer.getPrevIn(), dJdYhat, n_train, sW, rW, sb, rb,
                                         fullyConnectedLayer.getWeights(), fullyConnectedLayer.getBiases(), epoch)
                dJdYhat_ = customLayers[i].backward(dJdYhat)
                if customLayers[i] == convolutionalLayer:
                    convolutionalLayer.updateKernel(dJdYhat, 0.001)
                dJdYhat = dJdYhat_
        L_valid[epoch] /= 10
        L_train[epoch] /= 10
        print(L_train[epoch])
    plt.plot(np.arange(1, epoch + 1), L_train[1:epoch + 1], label="Training loss")
    plt.plot(np.arange(1, epoch + 1), L_valid[1:epoch + 1], color='g', label="Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.show()
    # plt.savefig("2.png")
    print("Training accuracy:", common.accuracy(n_train, X_train, Y_train, 70, len(customLayers), customLayers,
                                                crossEntropy), "%")
    print("Validation accuracy:", common.accuracy(n_valid, X_valid, Y_valid, 30, len(customLayers), customLayers,
                                                  crossEntropy), "%")
