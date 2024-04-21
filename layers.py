from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.stride_tricks import as_strided
import math

class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward(self, gradIn):
        pass


class InputLayer(Layer):
    # Input: dataIn, a NxD matrix
    # Output: None
    def __init__(self, dataIn):
        super().__init__()
        self.__meanX = dataIn.mean(axis=0)
        self.__stdX = np.std(dataIn, axis=0, ddof=1)
        for i in range(len(self.__stdX)):
            if self.__stdX[i] == 0:
                self.__stdX[i] = 1

    # Input: dataIn, a NxD matrix
    # Output: A NxD matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        ans = np.empty(shape=(np.shape(dataIn)))
        for i in range(len(self.__meanX)):
            ans[:, i] = (dataIn[:, i] - self.__meanX[i]) / self.__stdX[i]
        self.setPrevOut(ans)
        return ans

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass


class LinearLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()

    # Input: dataIn, a NxK matrix
    # Output: A NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        ans = dataIn.copy()
        self.setPrevOut(ans)
        return ans

    # Input: None
    # Output: A NxK matrix
    def gradient(self):
        return np.ones_like(self.getPrevIn())

    # Input: gradIn, a NxK matrix
    # Output: A NxK matrix
    def backward(self, gradIn):
        return gradIn * self.gradient()


class ReLuLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()

    # Input: dataIn, a NxK matrix
    # Output: A NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        ans = np.maximum(0, dataIn)
        self.setPrevOut(ans)
        return ans

    # Input: None
    # Output: A NxK matrix
    def gradient(self):
        prevIn = self.getPrevIn()
        return (prevIn >= 0) * 1

    # Input: gradIn, a NxK matrix
    # Output: A NxK matrix
    def backward(self, gradIn):
        return gradIn * self.gradient()


class LogisticSigmoidLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()

    # Input: dataIn, a NxK matrix
    # Output: A NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        ans = 1 / (1 + np.exp(-dataIn))
        self.setPrevOut(ans)
        return ans

    # Input: None
    # Output: A NxK matrix
    def gradient(self):
        prevOut = self.getPrevOut()
        return prevOut * (1 - prevOut)

    # Input: gradIn, a NxK matrix
    # Output: A NxK matrix
    def backward(self, gradIn):
        return gradIn * self.gradient()


class SoftmaxLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()

    # Input: dataIn, a NxK matrix
    # Output: A NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        accum = np.exp(dataIn - np.reshape(np.max(dataIn, axis=1), (-1, 1)))
        ans = accum / np.sum(accum, axis=1).reshape(-1, 1)
        self.setPrevOut(ans)
        return ans

    # Input: None
    # Output: A Nx(KxK) tensor
    def gradient(self):
        prevOut = self.getPrevOut()
        # Using vectorized implementation from https://stackoverflow.com/a/36280783 for faster training.
        J = - prevOut[..., None] * prevOut[:, None, :]  # off-diagonal Jacobian
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = prevOut * (1. - prevOut)  # diagonal
        return J

    # Input: gradIn, a NxK matrix
    # Output: A NxK matrix
    def backward(self, gradIn):
        return np.einsum('ij,ijk->ik', gradIn, self.gradient())


class TanhLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()

    # Input: dataIn, a NxK matrix
    # Output: A NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        ans = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
        self.setPrevOut(ans)
        return ans

    # Input: None
    # Output: A NxK matrix
    def gradient(self):
        prevOut = self.getPrevOut()
        return 1 - (prevOut ** 2)

    # Input: gradIn, a NxK matrix
    # Output: A NxK matrix
    def backward(self, gradIn):
        return gradIn * self.gradient()


class FullyConnectedLayer(Layer):
    # Input: sizeIn, the number of features of data coming in
    # Input: sizeOut, the number of features for the data coming out
    # Output: None
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        np.random.seed(0)
        self.__W = np.random.uniform(-10 ** (-4), 10 ** (-4), (sizeIn, sizeOut))
        self.__b = np.random.uniform(-10 ** (-4), 10 ** (-4), sizeOut)

    # Input: None
    # Output: The sizeIn x sizeOut weight matrix
    def getWeights(self):
        return self.__W

    # Input: The sizeIn x sizeOut weight matrix
    # Output: None
    def setWeights(self, weights):
        self.__W = weights

    # Input: None
    # Output: The 1 x sizeOut bias vector
    def getBiases(self):
        return self.__b

    # Input: The 1 x sizeOut bias vector
    # Output: None
    def setBiases(self, biases):
        self.__b = biases

    # Input: dataIn, a NxD data matrix
    # Output: A NxK data matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        ans = np.matmul(dataIn, self.__W) + self.__b
        self.setPrevOut(ans)
        return ans

    # Input: None
    # Output: A KxD matrix
    def gradient(self):
        return np.transpose(self.__W)

    # Input: gradIn, a NxK matrix
    # Output: A NxD matrix
    def backward(self, gradIn):
        return np.matmul(gradIn, self.gradient())

    # Input: gradIn, a NxK matrix
    # Intput: eta, a learning rate
    # Output: A NxD matrix
    def updateWeights(self, gradIn, eta=0.0001):
        dJdW = np.matmul(np.transpose(self.getPrevIn()), gradIn) / gradIn.shape[0]
        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]

        self.__W -= eta * dJdW
        self.__b -= eta * dJdb


class ConvolutionalLayer(Layer):
    def __init__(self):
        super().__init__()
        self.__K = np.random.uniform(-math.sqrt(6 / (3+3)), math.sqrt(6 / (3+3)), (3, 3))

    @staticmethod
    def starMulForward(a, b):
        # for loop implementation (original)
        # ans = np.zeros((a.shape[0], a.shape[1] - b.shape[0] + 1, a.shape[2] - b.shape[1] + 1, 3))
        # for i in range(a.shape[0]):
        #     for j in range(ans.shape[1]):
        #         for k in range(ans.shape[2]):
        #             for l in range(ans.shape[3]):
        #                 ans[i, j, k, l] = np.multiply(a[i, j:j+b.shape[0], k:k+b.shape[1], l], b[:, :, l]).sum()
        # return ans

        # Reference from https://stackoverflow.com/a/36280783 for faster calculation.
        Hout = a.shape[1] - b.shape[0] + 1
        Wout = a.shape[2] - b.shape[1] + 1
        a = as_strided(a, (a.shape[0], Hout, Wout, b.shape[0], b.shape[1]), a.strides[:3] + a.strides[1:])
        return np.tensordot(a, b, axes=2)

    def forward(self, dataIn):
        dataIn = np.array(dataIn)
        self.setPrevIn(dataIn)
        ans = self.starMulForward(dataIn, self.__K)
        return ans

    def gradient(self):
        pass

    @staticmethod
    def starMulBackward(a, b):
        Hout = a.shape[0] - b.shape[0] + 1
        Wout = a.shape[1] - b.shape[1] + 1

        a = as_strided(a, (Hout, Wout, b.shape[0], b.shape[1]), a.strides[:2] + a.strides[0:])
        return np.tensordot(a, b, axes=2)

    def backward(self, gradIn):
        return self.starMulForward(np.pad(gradIn, ((0, 0), (2, 2), (2, 2)), 'constant'), np.transpose(self.__K))

    def updateKernel(self, gradIn, eta=0.01):
        prevIn = self.getPrevIn()
        ans = np.empty_like(self.__K)
        for i in range(prevIn.shape[0]):
            ans[:, :] += self.starMulBackward(prevIn[i, :, :], gradIn[i, :, :])
        self.__K -= eta * ans


class MaxPoolLayer(Layer):
    def __init__(self, pool_size=(3, 3)):
        super().__init__()
        self.pool_size = pool_size
        self.cache = None

    def forward(self, X):
        # #       N is number of samples in input layer
        # #       H is number if rows of pixels
        # #       W is number of columns of pixels
        # N, H, W = X.shape
        # #       Height and Width of pooling window
        # HH, WW = self.pool_size
        # OH = H // HH
        # OW = W // WW
        # out = np.zeros((N, OH, OW))
        # self.cache = (X, out)
        #
        # for i in range(OH):
        #     for j in range(OW):
        #         out[:, i, j] = np.max(X[:, i * HH:(i + 1) * HH, j * WW:(j + 1) * WW], axis=(1, 2))
        #
        # return out

        # Reference from https://stackoverflow.com/a/74939121 for faster calculation.
        Hout = X.shape[1] // self.pool_size.shape[0]
        Wout = X.shape[2] // self.pool_size.shape[1]

        strides = (self.pool_size.shape[0] * X.shape[2], self.pool_size.shape[1], X.shape[2], 1)
        strides = tuple(i * X.itemsize for i in strides)
        ans = as_strided(X, (X.shape[0], Hout, Wout, self.pool_size.shape[0], self.pool_size.shape[1]), strides=strides)
        ans = np.max(ans, axis=(2, 3))
        return ans

    def gradient(self):
        pass

    # dout is the output from the forward method
    def backward(self, dout):
        X, out = self.cache
        N, H, W = X.shape
        HH, WW = self.pool_size
        OH = H // HH
        OW = W // WW
        dx = np.zeros_like(X)

        for i in range(OH):
            for j in range(OW):
                rl = (X[:, i * HH:(i + 1) * HH, j * WW:(j + 1) * WW] == np.max(
                    X[:, i * HH:(i + 1) * HH, j * WW:(j + 1) * WW], axis=(1, 2), keepdims=True))
                dx[:, i * HH:(i + 1) * HH, j * WW:(j + 1) * WW] += rl * (dout[:, i, j])[:, None, None]
        return dx


class FlattenedLayer(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, X):
        self.setPrevIn(X)
        self.input_shape = X.shape
        ans = X.reshape(X.shape[0], -1)
        self.setPrevOut(ans)
        return ans

    def gradient(self):
        pass

    def backward(self, dout):
        return dout.reshape(self.input_shape)


class DropoutLayer(Layer):
    def __init__(self, p=0.25):
        super().__init__()
        self.__p = p
        self.__mask = None

    def forward(self, dataIn):
        self.__mask = np.random.binomial(1, self.__p, size=dataIn.shape) / self.__p
        ans = dataIn * self.__mask
        return ans

    def gradient(self):
        pass

    def backward(self, gradIn):
        return gradIn * self.__mask

class SquaredError:
    # Input: Y is a NxK matrix of target values.
    # Input: Yhat is a NxK matrix of estimated values.
    # Output: A single floating point value.
    def eval(self, Y, Yhat):
        return np.mean((Y - Yhat) ** 2)

    # Input: Y is a NxK matrix of target values.
    # Input: Yhat is a NxK matrix of estimated values.
    # Output: A NxK matrix.
    def gradient(self, Y, Yhat):
        return -2 * (Y - Yhat)


class LogLoss:
    # Input: Y is a NxK matrix of target values.
    # Input: Yhat is a NxK matrix of estimated values.
    # Output: A single floating point value.
    def eval(self, Y, Yhat):
        return np.mean(-(Y * np.log(Yhat + np.finfo(float).eps) + (1 - Y) * np.log(1 - Yhat + np.finfo(float).eps)))

    # Input: Y is a NxK matrix of target values.
    # Input: Yhat is a NxK matrix of estimated values.
    # Output: A NxK matrix.
    def gradient(self, Y, Yhat):
        return -(Y - Yhat) / (Yhat * (1 - Yhat) + np.finfo(float).eps)


class CrossEntropy:
    # Input: Y is a NxK matrix of target values.
    # Input: Yhat is a NxK matrix of estimated values.
    # Output: A single floating point value.
    def eval(self, Y, Yhat):
        tmp = 1 / len(Y[0]) * np.sum(Y * np.log(Yhat + np.finfo(float).eps), axis=1)
        return -np.mean(tmp)

    # Input: Y is a NxK matrix of target values.
    # Input: Yhat is a NxK matrix of estimated values.
    # Output: A NxK matrix.
    def gradient(self, Y, Yhat):
        return -Y / (Yhat + np.finfo(float).eps)
