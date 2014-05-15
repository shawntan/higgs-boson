import pandas
import pandas as pd
import numpy as np
import math
def load_data(filename):
	df = pd.read_csv(filename,na_values=[-999.0])
	if 'Label' in df:
		df.reindex(np.random.permutation(df.index))
		labels = df['Label']
		labels[labels == 'b'] = 0
		labels[labels == 's'] = 1
		labels = np.asarray(labels.values,dtype=np.int8)
		weights = df['Weight'].values
		data = df.drop(['EventId','Weight','Label'],axis=1).astype(np.float64)
	else:
		labels = None
		weights = None
		data = df.drop(['EventId'],axis=1).astype(np.float64)

#	min_data = data.min()
#	print min_data[min_data >= 0]
#	data['DER_lep_eta_centrality'].apply(np.log)
#	data['DER_mass_jet_jet'] = data['DER_mass_jet_jet'].apply(np.log)
#	data['DER_deltaeta_jet_jet'] = data['DER_deltaeta_jet_jet'].apply(np.log)
#	data = (data - data.mean())/data.std()
	data = data.fillna(0).values
	return data,labels,weights,df['EventId']


