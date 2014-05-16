from data import *
import model
import pandas as pd
import cPickle as pickle
import numpy as np
import sys
if __name__ == '__main__':
	param_vals = pickle.load(open(sys.argv[1],'rb'))
	
	first_layer = np.sum(param_vals[0]**2,axis=1)
	sorted_features = sorted(zip(model.columns,first_layer),key=(lambda x: -x[1]))
	for label,score in sorted_features:
		print label,score



