from data import *
from model import *
import pandas as pd
import cPickle as pickle
if __name__ == '__main__':
	params_file = sys.argv[1]
	data,_,_,event_ids = load_data(sys.argv[2])
	idx = range(len(event_ids))
	df = pd.DataFrame(index=idx)
	print "Data Loaded."
	input_width = data.shape[1]
	X,_,output,parameters = build_network(input_width,512)
	param_vals = pickle.load(open(params_file,'rb'))
	for p,pv in zip(parameters,param_vals): p.set_value(pv)
	predict = theano.function(inputs=[X],outputs=output)
	df['EventId'] = event_ids
	df['probs'] = predict(data)
	df['Class'] = ['b'] * len(event_ids)
	df['Class'][df.probs > 0.5] = 's'
	df['RankOrder'] = df.probs.rank(ascending=False,method='first').astype(int)
	df.to_csv('data/submission.csv',cols=['EventId','RankOrder','Class'],index=False)
