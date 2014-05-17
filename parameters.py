from data import *
import pandas as pd
import cPickle as pickle
import numpy as np
import sys
if __name__ == '__main__':
	model = pickle.load(open(sys.argv[1],'rb'))
	param_vals = model['parameters']
	first_layer = np.sum(param_vals[0]**2,axis=1)
	sorted_features = sorted(zip(model['feature_names'],first_layer),key=(lambda x: -x[1]))
	for label,score in sorted_features: print label,score



