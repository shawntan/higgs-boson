import math
import theano
import theano.tensor as T
import numpy as np

from theano_toolkit.parameters import Parameters
from theano_toolkit import updates

import data
import model

def ams(Y_hat, Y, W, b_r=10):
    s = T.sum(W * Y * Y_hat)
    b = T.sum(W * (1-Y) * Y_hat)
    ams_score = T.sqrt(2*((s + b + b_r) * (T.log(1 + s) - T.log(b + b_r))) - s))
    return ams_score


def get_train_test_fn(P, data_X, data_W, data_Y):

    X = T.matrix('X')
    Y = T.ivector('Y')
    W = T.vector('W')

    f = model.build(P,
        input_size=data_X.get_value().shape[1],
        hidden_sizes=[256, 128, 64, 32]
    )
    W_s = T.sum(T.switch(data_Y, data_W, 0))
    W_b = T.sum(T.switch(data_Y, 0, data_W))
    W_s_batch = T.sum(T.switch(Y, W, 0))
    W_b_batch = T.sum(T.switch(Y, 0, W))

    W_hat = T.switch(Y,
                     W * W_s / W_s_batch,
                     W * W_b / W_b_batch)

    output = f(X)
    test_output = f(X, test=True)
    soft_ams = ams(output, Y, W_hat)
    discrete_ams = ams(output > 0.5, Y, W_hat)
    parameters = P.values()
    gradients = T.grad(-soft_ams, wrt=parameters)

    idx = T.iscalar('idx')
    batch_size = T.iscalar('batch_size')
    train = theano.function(
        inputs=[idx, batch_size],
        outputs=[soft_ams, discrete_ams],
        updates=updates.adam(parameters, gradients, learning_rate=5e-4),
        givens={
            X: data_X[idx * batch_size: (idx + 1) * batch_size],
            W: data_W[idx * batch_size: (idx + 1) * batch_size],
            Y: data_Y[idx * batch_size: (idx + 1) * batch_size],
        }
    )

    test = theano.function(
        inputs=[X, W, Y],
        outputs=[ams(test_output > 0.5, Y, W_hat),
                 ams(test_output > 0.7, Y, W_hat),
                 ams(test_output > 0.9, Y, W_hat),
                 ams(test_output > 0.99, Y, W_hat)]
    )
    print "Compilation done."

    return train, test

if __name__ == "__main__":
    batch_size = 256
    validation = 0.1

    all_X, all_W, all_Y = data.load('data/training.csv')
    validation_count = int(math.ceil(all_X.shape[0] * validation))
    train_X, train_W, train_Y = (all_X[:-validation_count],
                                 all_W[:-validation_count],
                                 all_Y[:-validation_count])

    valid_X, valid_W, valid_Y = (all_X[-validation_count:],
                                 all_W[-validation_count:],
                                 all_Y[-validation_count:])

    P = Parameters()
    data_X = theano.shared(train_X)
    data_W = theano.shared(train_W)
    data_Y = theano.shared(train_Y)

    train, test = get_train_test_fn(P, data_X, data_W, data_Y)
    batches = int(math.ceil(train_X.shape[0] / float(batch_size)))
    best_score = -np.inf
    for epoch in xrange(20):
        for i in xrange(batches):
            train(i, batch_size)
        scores = test(valid_X, valid_W, valid_Y)
        print scores,
        if scores[0] > best_score :
            P.save('model.pkl')
            best_score = scores[0]
            print "Saved."
        else:
            print
        s = np.random.randint(1337)
        np.random.seed(s)
        np.random.shuffle(train_X)
        np.random.seed(s)
        np.random.shuffle(train_W)
        np.random.seed(s)
        np.random.shuffle(train_Y)
        data_X.set_value(train_X)
        data_W.set_value(train_W)
        data_Y.set_value(train_Y)

