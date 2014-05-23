import pandas as pd
import numpy  as np
df = pd.read_csv('data/training.csv',na_values=[-999.0])
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
weights = df['Weight']
data = df[columns]
for col in data:
	sel_cols = pd.notnull(data[col])
	D = data[col][sel_cols]
	W = weights[sel_cols]
	Wmean = (D*(W/W.sum())).sum()
	#print "Mean:", Wmean
	Wstd = np.sqrt(((W/W.sum()) * (D - Wmean)**2).sum())
	print "Std:", Wstd


