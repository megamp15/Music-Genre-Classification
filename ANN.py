import numpy as np
import layers
import common_ann
import matplotlib.pyplot as plt


def init():
    dataset_train, dataset_test = common_ann.GTZAN_dataset()
    data_train, n_train, X_train, Y_train = common_ann.read_file("", dataset_train)
    data_valid, n_valid, X_valid, Y_valid = common_ann.read_file("", dataset_test)
    inputLayer = layers.InputLayer(X_train)
    X_train = inputLayer.forward(X_train)
    X_valid = inputLayer.forward(X_valid)
    num_layer = 6
    sW = [None] * num_layer
    rW = [None] * num_layer
    sb = [None] * num_layer
    rb = [None] * num_layer
    customLayers = [None] * num_layer
    customLayers[0] = layers.DropoutLayer()
    customLayers[1], sW[1], rW[1], sb[1], rb[1] = common_ann.ADAM_init(X_train.shape[1], 850)
    customLayers[2] = layers.ReLuLayer()
    customLayers[3] = layers.DropoutLayer()
    customLayers[4], sW[4], rW[4], sb[4], rb[4] = common_ann.ADAM_init(850, 10)
    customLayers[5] = layers.SoftmaxLayer()
    return n_train, X_train, Y_train, data_train, n_valid, X_valid, Y_valid, data_valid, sW, rW, sb, rb,\
        num_layer, customLayers


def train(n_train, X_train, Y_train, data_train, n_valid, X_valid, Y_valid, data_valid, sW, rW, sb, rb,
          num_layer, customLayers):
    crossEntropy = layers.CrossEntropy()
    L_train = np.zeros(101)
    L_valid = np.zeros(101)
    batch_size_train = (len(X_train) - 1) // 10 + 1
    batch_size_valid = (len(X_valid) - 1) // 10 + 1
    for epoch in range(1, 101):
        if epoch % 10 == 0:
            print("epoch:", epoch)
        for m_batch in range(10):
            begin_valid = m_batch * batch_size_valid
            end_valid = min((m_batch + 1) * batch_size_valid, len(X_valid))
            Yhat_valid, l_valid = common_ann.forward(X_valid[begin_valid: end_valid], Y_valid[begin_valid: end_valid],
                                                 num_layer, customLayers, crossEntropy)
            L_valid[epoch] += l_valid
            begin_train = m_batch * batch_size_train
            end_train = min((m_batch + 1) * batch_size_train, len(X_train))
            Yhat_train, l_train = common_ann.forward(X_train[begin_train:end_train], Y_train[begin_train:end_train],
                                                 num_layer, customLayers, crossEntropy)
            L_train[epoch] += l_train
            # backward
            dJdYhat = crossEntropy.gradient(Y_train[begin_train:end_train], Yhat_train)
            for i in reversed(range(len(customLayers))):
                if i == 1 or i == 4:
                    common_ann.ADAM_learning(customLayers[i].getPrevIn(), dJdYhat, n_train, sW[i], rW[i], sb[i], rb[i],
                                         customLayers[i].getWeights(), customLayers[i].getBiases(), epoch)
                dJdYhat = customLayers[i].backward(dJdYhat)
        L_valid[epoch] /= 10
        L_train[epoch] /= 10

    print("Training accuracy:", common_ann.accuracy(n_train, X_train, Y_train, batch_size_train, num_layer, customLayers,
                                                crossEntropy), "%")
    print("Validation accuracy:", common_ann.accuracy(n_valid, X_valid, Y_valid, batch_size_valid, num_layer, customLayers,
                                                  crossEntropy), "%")

    plt.plot(np.arange(1, epoch + 1), L_train[1:epoch + 1], label="Training loss")
    plt.plot(np.arange(1, epoch + 1), L_valid[1:epoch + 1], color='g', label="Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.title('Epoch vs Loss')
    plt.show()


if __name__ == '__main__':
    n_train, X_train, Y_train, data_train, n_valid, X_valid, Y_valid, data_valid, sW, rW, sb, rb, num_layer, \
        customLayers = init()
    train(n_train, X_train, Y_train, data_train, n_valid, X_valid, Y_valid, data_valid, sW, rW, sb, rb, num_layer,
          customLayers)
