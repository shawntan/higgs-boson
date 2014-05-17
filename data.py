import pandas
import pandas as pd
import numpy as np
import math

columns = [
		"DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis",
		"DER_pt_h","DER_deltaeta_jet_jet","DER_mass_jet_jet",
		"DER_prodeta_jet_jet","DER_deltar_tau_lep","DER_pt_tot","DER_sum_pt",
		"DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality",
		"PRI_tau_pt","PRI_tau_eta","PRI_lep_pt","PRI_lep_eta",
		"PRI_met","PRI_met_sumet","PRI_jet_num",
		"PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi",
		"PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi",
		"PRI_jet_all_pt",
		"PRI_lep_phi","PRI_met_phi","PRI_tau_phi"
	]

def momentum_vec(p_T,phi,eta=None):
	x = p_T * np.cos(phi)/p_T
	y = p_T * np.sin(phi)/p_T
	if eta is None:
		return x,y
	else:
		z = p_T * np.sinh(eta)/p_T
		return x,y,z


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
#	data['DER_mass_jet_jet_log'] = data['DER_mass_jet_jet'].apply(np.log)
#	data['DER_deltaeta_jet_jet'] = data['DER_deltaeta_jet_jet'].apply(np.log)
	data['DER_mass_MMC'] = data['DER_mass_MMC'].apply(np.log)
#	data['DER_mass_vis_log'] = data['DER_mass_vis'].apply(np.log)
#	data['DER_mass_transverse_met_lep_log'] = data['DER_mass_transverse_met_lep'].apply(np.log)
	data = (data - data.mean())/data.std()
	for p in ['tau','lep','jet_leading','jet_subleading']:
		data['%s_x'%p], data['%s_y'%p], data['%s_z'%p] =\
			momentum_vec(data['PRI_%s_pt'%p],data['PRI_%s_phi'%p],data['PRI_%s_eta'%p])
	data['met_x'],data['met_y'] = momentum_vec(data['PRI_met'],data['PRI_met_phi'])


	for col in columns:
		nan_rows = pd.isnull(data[col])
		if nan_rows.sum() > 0:
			data["ISNA_%s"%col] = 0
			data["ISNA_%s"%col][nan_rows] = 1
	for i in range(4):
		data["jet_%d"%i] = 0
		data["jet_%d"%i][data.PRI_jet_num == i] = 1


	return data.fillna(0).values,labels,weights,df['EventId'], [ n for n in data ]


