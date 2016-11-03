import math
import theano
import theano.tensor as T
import numpy as np
import feedforward
from theano_toolkit import utils as U
def build(P, input_size, hidden_sizes):

    def activation(X):
        mask = U.theano_rng.binomial(size=X.shape, p=0.5)
        return T.switch(mask, T.nnet.relu(X), 0)

    classifier = feedforward.build_classifier(
        P, name="classifier",
        input_sizes=[input_size],
        hidden_sizes=hidden_sizes,
        output_size=1,
        initial_weights=feedforward.relu_init,
        output_initial_weights=lambda x,y: np.zeros((x,y)),
        activation=activation,
        output_activation=T.nnet.sigmoid)

    def predict(X):
        return classifier([X])[:, 0]
    return predict


