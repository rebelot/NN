import numpy as np
import matplotlib.pyplot as plt
from NNpro import NN, LinearLayer, ActivationLayer, DropoutLayer, BatchNormLayer, CompositeLayer, ConvolutionLayer, PoolingLayer, AdapterLayer, BranchLayer, JoinLayer, BranchLayer, JoinLayer
from timeit import default_timer as timer

np.random.seed(1)

def XOR_test():
    X = np.random.uniform(-1, 1, size=(2, 1000))
    Y = (X.prod(axis=0) > 0).astype(int).reshape(1, X.shape[1])
    Y = np.concatenate((Y, abs(1 - Y)), axis=0)
    # X = np.concatenate((X, Y), axis=0)

    layers = [
        BatchNormLayer(X.shape[0]),
        LinearLayer(10),
        ActivationLayer(1),
        BranchLayer(1, name='1'),
        LinearLayer(10),
        ActivationLayer(1),
        LinearLayer(10),
        JoinLayer(1, branch_name='1'),
        ActivationLayer(1),
        BranchLayer(1, name='2'),
        LinearLayer(10),
        ActivationLayer(1),
        LinearLayer(10),
        JoinLayer(1, branch_name='2'),
        ActivationLayer(1),
        CompositeLayer(2, initializer="xavier", activation="sigmoid"),
    ]

    nn_settings = {"loss_function": "mse", "optimizer": 'adam'}

    training_settings = {
        "dev_split": 0.2,
        "batch_size": 0,
        "epochs": 10000,
        "learning_rate": 0.01,
        "lambd": 0.2,
        "draw": True,
    }

    nn = NN(*layers)
    nn.compile(**nn_settings)
    nn.init_parameters(X.shape)
    nn.init_optimizer()
    nn.info(v=1)
    start = timer()
    t = nn.train(X, Y, **training_settings)
    stop = timer()
    print(stop - start)
    return nn


def CONV_test():
    from PIL import Image

    cats = [f'../train/cat.{i}.jpg' for i in range(500)]
    dogs = [f'../train/dog.{i}.jpg' for i in range(500)]


    catimgs = []
    dogimgs = []
    for cat, dog in zip(cats,dogs):
        c = Image.open(cat).resize((100,100))
        d = Image.open(dog).resize((100,100))
        ca = np.einsum('hwc->chw', np.asarray(c))/255
        da = np.einsum('hwc->chw', np.asarray(d))/255
        ca = (ca - ca.mean(axis=(1,2)).reshape(-1,1,1)) / ca.std(axis=(1,2)).reshape(-1,1,1)
        da = (da - da.mean(axis=(1,2)).reshape(-1,1,1)) / da.std(axis=(1,2)).reshape(-1,1,1)
        catimgs.append(ca)
        dogimgs.append(da)

    catimgs = np.array(catimgs)
    dogimgs = np.array(dogimgs)

    X = np.concatenate((catimgs, dogimgs), axis=0)
    Y = np.concatenate((np.ones(500), np.zeros(500))).reshape(-1, 1)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx].T

    blacks = np.ones((500,2,5,5))
    blacks[:,0] *= -1
    whites = np.ones((500,2,5,5))
    X = np.r_[blacks, whites]
    Y = np.concatenate((np.ones(500), np.zeros(500))).reshape(-1, 1).T


    layers = [
            ConvolutionLayer((1,3,3), mode='same', initializer='he_uniform'),
            ActivationLayer(1),
            # ConvolutionLayer((64,3,3), mode='same', initializer='he_uniform'),
            # ActivationLayer(1),
            # PoolingLayer((1,2,2), stride=2),
            # ConvolutionLayer((128,3,3), mode='same', initializer='he_uniform'),
            # ActivationLayer(1),
            # PoolingLayer((1,2,2), stride=2, operation='max'),
            AdapterLayer(1),
            # CompositeLayer(10, initializer="xavier_uniform", activation='tanh'),
            CompositeLayer(1, initializer="xavier_uniform", activation="sigmoid"),
    ]

    nn_settings = {"loss_function": "logloss", "optimizer": 'adam'}

    training_settings = {
        "dev_split": 0.2,
        "batch_size": 2**7,
        "epochs": 100,
        "learning_rate": 0.001,
        "lambd": 0,
        "draw": True,
        "every": 10
    }

    nn = NN(*layers)
    nn.compile(**nn_settings)
    nn.init_parameters(X[0].shape)
    nn.init_optimizer()
    nn.info()
    start = timer()
    try:
        t = nn.train(X, Y, **training_settings)
    except KeyboardInterrupt:
        return nn
    stop = timer()
    print(stop - start)
    return nn


if __name__ == "__main__":
    nn = CONV_test()
    # nn = XOR_test()
