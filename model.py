import math
import theano
import theano.tensor as T
import numpy as np
import feedforward
from theano_toolkit import utils as U
def build(P, input_size, hidden_sizes):
    test_time = False
    def activation(X):
        global test_time
        if not test_time:
            mask = U.theano_rng.binomial(size=X.shape, p=0.5)
            return T.switch(mask, T.nnet.relu(X), 0)
        else:
            return 0.5 * T.nnet.relu(X)


    classifier = feedforward.build_classifier(
        P, name="classifier",
        input_sizes=[input_size],
        hidden_sizes=hidden_sizes,
        output_size=1,
        initial_weights=feedforward.relu_init,
        output_initial_weights=lambda x,y: np.zeros((x,y)),
        activation=activation,
        output_activation=T.nnet.sigmoid)

    def predict(X, test=False):
        global test_time
        test_time = test
        return classifier([X])[:, 0]
    return predict

