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


def load(filename):
    df = pd.read_csv(filename)
    Y = np.array(df.ix[:, 'Label']) == 's'
    W = np.array(df.ix[:, 'Weight']).astype(np.float32)
    X = np.array(df.ix[:,
        (df.columns != 'EventId') & 
        (df.columns != 'Label') & 
        (df.columns != 'Weight')]).astype(np.float32)
    return X, W, Y.astype(np.int32)

