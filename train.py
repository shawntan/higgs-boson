import math
import theano
import theano.tensor as T

from theano_toolkit.parameters import Parameters
from theano_toolkit import updates

import data
import model

def get_train_test_fn(train_X, train_W, train_Y):
    P = Parameters()

    X = T.matrix('X')
    Y = T.ivector('Y')
    W = T.vector('W')

    data_X = theano.shared(train_X)
    data_W = theano.shared(train_W)
    data_Y = theano.shared(train_Y)

    f = model.build(P,
        input_size=train_X.shape[1],
        hidden_sizes=[512]
    )
    W_hat = W * T.sum(data_W) / T.sum(W)
    output = f(X)
    soft_ams = ams(output, Y, W_hat)
    discrete_ams = ams(output > 0.5, Y, W_hat)
    parameters = P.values()
    gradients = T.grad(-soft_ams, wrt=parameters)

    idx = T.iscalar('idx')
    batch_size = T.iscalar('batch_size')
    train = theano.function(
        inputs=[idx, batch_size],
        outputs=[soft_ams, discrete_ams],
        updates=updates.adam(parameters, gradients),
        givens={
            X: data_X[idx * batch_size: (idx + 1) * batch_size],
            W: data_W[idx * batch_size: (idx + 1) * batch_size],
            Y: data_Y[idx * batch_size: (idx + 1) * batch_size],
        }
    )

    test = theano.function(
        inputs=[X, W, Y],
        outputs=discrete_ams,
    )


    return train, test

def ams(Y_hat, Y, W, b_r=10):
    s = T.sum(W * Y * Y_hat)
    b = T.sum(W * (1-Y) * Y_hat)
    ams_score = T.sqrt(2*((s + b + b_r) * T.log(1 + s/(b + b_r)) - s))
    return ams_score


if __name__ == "__main__":
    batch_size = 128
    validation = 0.1
    all_X, all_W, all_Y = data.load('data/training.csv')
    validation_count = int(math.ceil(all_X.shape[0] * validation))
    train_X, train_W, train_Y = (all_X[:-validation_count],
                                 all_W[:-validation_count],
                                 all_Y[:-validation_count])

    valid_X, valid_W, valid_Y = (all_X[-validation_count:],
                                 all_W[-validation_count:],
                                 all_Y[-validation_count:])

    train, test = get_train_test_fn(train_X, train_W, train_Y)
    batches = int(math.ceil(train_X.shape[0] / float(batch_size)))
    for i in xrange(batches):
        print train(i, batch_size)
    test(valid_X, valid_W, valid_Y)
