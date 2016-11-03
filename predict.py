import pandas as pd
import cPickle as pickle
import theano
import theano.tensor as T
import numpy as np
import sys

import data
import model
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates


if __name__ == '__main__':
    model_filename = sys.argv[1]
    test_filename = sys.argv[2]
    train_filename = sys.argv[3]
    P = Parameters()
    data_X, df = data.load_test(test_filename, train_filename)
    f = model.build(P,
        input_size=data_X.shape[1],
        hidden_sizes=[256, 128, 64, 32]
    )
    X = T.matrix('X')
    predict = theano.function(
        inputs=[X],
        outputs=f(X, test=True) > 0.5,
    )
    P.load(model_filename)
    output = predict(data_X) 
    print data_X.shape
    print output.shape
    print df.values.shape
    df['probs'] = predict(data_X)
    df['Class'] = 'b'
    df['Class'][df.probs > 0.5] = 's'
    df['RankOrder'] = df.probs.rank(ascending=False,method='first').astype(int)
    df.to_csv('data/submission.csv', cols=['EventId','RankOrder','Class'], index=False)
