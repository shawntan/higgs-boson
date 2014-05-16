import pandas
import pandas as pd
import numpy as np
import math

columns = [
		"DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis",
		"DER_pt_h","DER_deltaeta_jet_jet","DER_mass_jet_jet",
		"DER_prodeta_jet_jet","DER_deltar_tau_lep","DER_pt_tot","DER_sum_pt",
		"DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality",
		"PRI_tau_pt","PRI_tau_eta","PRI_tau_phi","PRI_lep_pt","PRI_lep_eta",
		"PRI_lep_phi","PRI_met","PRI_met_phi","PRI_met_sumet","PRI_jet_num",
		"PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi",
		"PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi",
		"PRI_jet_all_pt"
	]

def load_data(filename):
	df = pd.read_csv(filename,na_values=[-999.0])
	data = df[columns].astype(np.float32)
	if 'Label' in df:
		df.reindex(np.random.permutation(df.index))
		labels = df['Label']
		labels[labels == 'b'] = 0
		labels[labels == 's'] = 1
		labels = np.asarray(labels.values,dtype=np.int8)
		weights = df['Weight'].values
		#data = df.drop(['EventId','Weight','Label'],axis=1).astype(np.float64)
	else:
		labels = None
		weights = None
		#data = df.drop(['EventId'],axis=1).astype(np.float64)

#	min_data = data.min()
#	print min_data[min_data >= 0]
#	data['DER_lep_eta_centrality'].apply(np.log)
#	data['DER_mass_jet_jet'] = data['DER_mass_jet_jet'].apply(np.log)
#	data['DER_deltaeta_jet_jet'] = data['DER_deltaeta_jet_jet'].apply(np.log)
	data = (data - data.mean())/data.std()
	data = data.fillna(0).values
	return data,labels,weights,df['EventId']

