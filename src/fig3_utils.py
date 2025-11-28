import numpy as np
import plotly.io as pio

import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn 
import os
from matplotlib import cm as colours 
from colorama import Fore, Back, Style
from scipy.stats import sem
import joblib
import umap.umap_ as umap
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_samples
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from plotly.subplots import make_subplots
import pickle 
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from sklearn.utils import shuffle
numtoPhon = {1:'a', 2:'ae', 3:'i', 4:'u', 5:'b', 6:'p', 7:'v', 8:'g', 9:'k'}
phoneme_group = {
        'a': 'low', 'ae': 'low', 'i': 'high', 'u': 'high',
        'b': 'labial', 'p': 'labial', 'v': 'labial',
        'g': 'dorsal', 'k': 'dorsal'
    }

class PCA_noCenter(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = self._get_components(X, S)
        self.components_ = Vt[:k].T
        self.explained_variance_ = S**2
        return self

    def transform(self, X):
        return X @ self.components_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def _get_components(self, X, S):
        if self.n_components is None or self.n_components >= min(X.shape):
            print("n_components is None or greater than the number of features/samples. Using n_components = min(X.shape)")
            return min(X.shape)
        elif self.n_components < 1:
            cum_var = np.cumsum(S**2) / np.sum(S**2)
            return np.argmax(cum_var >= self.n_components) + 1
        else:
            return int(self.n_components)
        



def phoneme_type(phoneme):
    
    consonants = ['b', 'p', 'v', 'g', 'k']
    vowels = ['a', 'ae', 'i', 'u']
    
    if phoneme.lower() in consonants:
        return "Consonant"
    elif phoneme.lower() in vowels:
        return "Vowel"
    else:
        raise ValueError("Phoneme not in categories")

def get_position_index(position: str) -> int:
    position_map = {'p1': 0, 'p2': 1, 'p3': 2}
    if position not in position_map:
        raise ValueError(f"Invalid position: {position}")
    return position_map[position]



def fetch_patient_data(pkl_path):
    
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            patient_data = pickle.load(f)
        print("Loaded patient_data from pickle.")
    else:
        path = r"C:\Users\Nabiya\Box\Academic-Duke\CoganLab\Summer 2024\intraop\ieeg_data_intraop\Zac_intraop_data"
        patients = ['S14', 'S22', 'S23', 'S26', 'S33']
        positions = ['p1', 'p2', 'p3']

        patient_data = {}

        for patient in patients:
            patient_data[patient] = {}
            for position in positions:
                patient_data[patient][position] = {'Kumar': {}, 'MFA': {}}

                data_kumar = sp.io.loadmat(f"{path}\\{patient}\\{patient}_HG_{position}_sigChannel_goodTrials.mat")
                data_mfa = sp.io.loadmat(f"{path}\\{patient}\\{patient}_HG_{position}_sigChannel_goodTrials_MFA.mat")

                patient_data[patient][position]['Kumar'] = {
                    'hg_trace': data_kumar['hgTrace'],
                    'hg_map': data_kumar['hgMap'],
                    'phon_seq': data_kumar['phonSeqLabels']
                }

                patient_data[patient][position]['MFA'] = {
                    'hg_trace': data_mfa['hgTrace'],
                    'hg_map': data_mfa['hgMap'],
                    'phon_seq': data_mfa['phonSeqLabels']
                }

        # Save to pickle
        with open(pkl_path, "wb") as f:
            pickle.dump(patient_data, f)

        print("Processed and saved patient_data to pickle.")
    return patient_data



def get_training_data(patient_data, patient_name, position, method):
    if position == 'all':
        X_train = np.concatenate(
            (
            patient_data[patient_name]['p1'][method]['hg_trace'], 
            patient_data[patient_name]['p2'][method]['hg_trace'], 
            patient_data[patient_name]['p3'][method]['hg_trace']
            ), 
            axis=0
        )
        y_train = np.concatenate(
            (
            patient_data[patient_name]['p1'][method]['phon_seq'][:, 0],
            patient_data[patient_name]['p2'][method]['phon_seq'][:, 1],
            patient_data[patient_name]['p3'][method]['phon_seq'][:, 2]
            ), 
            axis=0
        )
        
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train_positions = np.concatenate(
            [
            ['p1']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 0]),
            ['p2']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 1]),
            ['p3']*len(patient_data[patient_name]['p1'][method]['phon_seq'][:, 2])
            ],
            axis=0)
    else:
        X_train = patient_data[patient_name][position][method]['hg_trace']
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = patient_data[patient_name][position][method]['phon_seq'][:, get_position_index(position)]
        y_train_positions = np.array([position]*len(y_train))
    
    y_train_phon = np.array([numtoPhon[i] for i in y_train])
    y_ph_group = np.array([phoneme_group[i] for i in y_train_phon])
    y_ph_type = np.array([phoneme_type(i) for i in y_train_phon])
    Xys_df = pd.DataFrame({
        'X_train': list(X_train),  # each row is a feature vector
        'y_train_num': y_train,
        'y_train_phon': y_train_phon,
        'y_train_phontype': y_ph_type,
        'y_train_phongrp': y_ph_group,
        'y_train_pos': y_train_positions,
        'Method': [method] * len(y_train),
    })
    Xys_df['Patient'] = patient_name
    return Xys_df


def run_tsne(Xys_df, perp, pcacomp, numrun): 
    tsne_model = TSNE(n_components=2, init='pca', perplexity=perp, n_jobs=-1)
    pca_model = PCA_noCenter(n_components=pcacomp)
    
    X_train = np.stack(Xys_df['X_train'].values)
    y_train = Xys_df['y_train_num'].to_numpy()
    y_train_pos = Xys_df['y_train_pos'].to_numpy()
    y_train_type = Xys_df['y_train_phontype'].to_numpy()
    y_train_phgrp = Xys_df['y_train_phongrp'].to_numpy()
    
    X_pca = pca_model.fit_transform(X_train)
    X_tsne = tsne_model.fit_transform(X_pca)
    
    tsne_df = pd.DataFrame(X_tsne, columns=['tsne-1', 'tsne-2'])
    tsne_df['Phoneme'] = Xys_df['y_train_phon']
    tsne_df['Phoneme_Position'] = Xys_df['y_train_pos']
    tsne_df['Phoneme_Type'] = Xys_df['y_train_phontype']
    tsne_df['Phoneme_Group'] = Xys_df['y_train_phongrp']
    tsne_df['Perplexity'] = perp
    tsne_df['KL_Divergence'] = tsne_model.kl_divergence_
    tsne_df['Silhoutte_Score_Phon'] = np.mean(np.array(silhouette_samples(X_tsne, y_train))[(np.array(silhouette_samples(X_tsne, y_train)) > 0)])
    if tsne_df['Phoneme_Position'].nunique() > 2:
        tsne_df['Silhoutte_Score_PhonPos'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_pos))[(np.array(silhouette_samples(X_tsne, y_train_pos)) > 0)])
    tsne_df['Silhoutte_Score_PhonType'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_type))[(np.array(silhouette_samples(X_tsne, y_train_type)) > 0)])
    tsne_df['Silhoutte_Score_PhonGrp'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_phgrp))[(np.array(silhouette_samples(X_tsne, y_train_phgrp)) > 0)])
    tsne_df['Patient'] = Xys_df['Patient'].iloc[0]
    tsne_df['Method'] = Xys_df['Method'].iloc[0] 
    tsne_df['#Run'] = numrun
    if tsne_df['Phoneme_Position'].nunique() > 2:
        tsne_df = tsne_df[['#Run', 'tsne-1', 'tsne-2', 'Phoneme', 'Phoneme_Position', 'Phoneme_Type',
                           'Phoneme_Group', 'Perplexity', 'KL_Divergence','Silhoutte_Score_Phon',
                           'Silhoutte_Score_PhonPos', 'Silhoutte_Score_PhonType',
                           'Silhoutte_Score_PhonGrp', 'Patient', 'Method']]
    else:
        tsne_df = tsne_df[['#Run', 'tsne-1', 'tsne-2', 'Phoneme', 'Phoneme_Position', 'Phoneme_Type',
                           'Phoneme_Group', 'Perplexity', 'KL_Divergence','Silhoutte_Score_Phon',
                           'Silhoutte_Score_PhonType', 'Silhoutte_Score_PhonGrp','Patient', 'Method']]
    return tsne_df
    
def run_tsneShuffledY(Xys_df, perp, pcacomp, numrun): 
    tsne_model = TSNE(n_components=2, init='pca', perplexity=perp, n_jobs=-1)
    pca_model = PCA_noCenter(n_components=pcacomp)
    
    X_train = np.stack(Xys_df['X_train'].values)
    y_train = Xys_df['y_train_num'].to_numpy()
    # to create a chance silhouette distribution, we will shuffle the labels
    y_trainChance = shuffle(y_train, random_state=numrun)
    y_train_pos = Xys_df['y_train_pos'].to_numpy()
    y_train_type = Xys_df['y_train_phontype'].to_numpy()
    y_train_phgrp = Xys_df['y_train_phongrp'].to_numpy()
    
    X_pca = pca_model.fit_transform(X_train)
    X_tsne = tsne_model.fit_transform(X_pca)
    
    tsne_df = pd.DataFrame(X_tsne, columns=['tsne-1', 'tsne-2'])
    tsne_df['Phoneme'] = Xys_df['y_train_phon']
    tsne_df['Phoneme_Position'] = Xys_df['y_train_pos']
    tsne_df['Phoneme_Type'] = Xys_df['y_train_phontype']
    tsne_df['Phoneme_Group'] = Xys_df['y_train_phongrp']
    tsne_df['Perplexity'] = perp
    tsne_df['KL_Divergence'] = tsne_model.kl_divergence_
    tsne_df['Silhoutte_Score_Phon'] = np.mean(np.array(silhouette_samples(X_tsne, y_train))[(np.array(silhouette_samples(X_tsne, y_train)) > 0)])
    tsne_df['Silhouette_Score_PhonChance'] = np.mean(np.array(silhouette_samples(X_tsne, y_trainChance))[(np.array(silhouette_samples(X_tsne, y_trainChance)) > 0)])
    if tsne_df['Phoneme_Position'].nunique() > 2:
        tsne_df['Silhoutte_Score_PhonPos'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_pos))[(np.array(silhouette_samples(X_tsne, y_train_pos)) > 0)])
    tsne_df['Silhoutte_Score_PhonType'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_type))[(np.array(silhouette_samples(X_tsne, y_train_type)) > 0)])
    tsne_df['Silhoutte_Score_PhonGrp'] = np.mean(np.array(silhouette_samples(X_tsne, y_train_phgrp))[(np.array(silhouette_samples(X_tsne, y_train_phgrp)) > 0)])
    tsne_df['Patient'] = Xys_df['Patient'].iloc[0]
    tsne_df['Method'] = Xys_df['Method'].iloc[0] 
    tsne_df['#Run'] = numrun
    if tsne_df['Phoneme_Position'].nunique() > 2:
        tsne_df = tsne_df[['#Run', 'tsne-1', 'tsne-2', 'Phoneme', 'Phoneme_Position', 'Phoneme_Type',
                           'Phoneme_Group', 'Perplexity', 'KL_Divergence',
                           'Silhoutte_Score_Phon', 'Silhouette_Score_PhonChance',
                           'Silhoutte_Score_PhonPos', 'Silhoutte_Score_PhonType',
                           'Silhoutte_Score_PhonGrp', 'Patient', 'Method']]
    else:
        tsne_df = tsne_df[['#Run', 'tsne-1', 'tsne-2', 'Phoneme', 'Phoneme_Position', 'Phoneme_Type',
                           'Phoneme_Group', 'Perplexity', 'KL_Divergence',
                           'Silhoutte_Score_Phon', 'Silhouette_Score_PhonChance',
                           'Silhoutte_Score_PhonType', 'Silhoutte_Score_PhonGrp','Patient', 'Method']]
    return tsne_df
    

def run_plottly(tsne_df, huere):
    pio.templates.default = "none"
    catorder = None
    if huere =='Phoneme':
        silscore = tsne_df['Silhoutte_Score_Phon'].iloc[0]
        plette = px.colors.qualitative.Dark24_r
        catorder = {"Phoneme":['a', 'ae', 'b', 'g', 'i', 'k', 'p', 'u', 'v']} 
    elif huere == 'Phoneme_Position':
        silscore = tsne_df['Silhoutte_Score_PhonPos'].iloc[0]
        plette = px.colors.qualitative.Set1
    elif huere == 'Phoneme_Type':
        silscore = tsne_df['Silhoutte_Score_PhonType'].iloc[0]
        plette = px.colors.qualitative.Pastel
    elif huere == 'Phoneme_Group':
        silscore = tsne_df['Silhoutte_Score_PhonGrp'].iloc[0]
        plette = px.colors.qualitative.G10
    else:
        raise ValueError("Invalid hue parameter. Choose from 'Phoneme', 'Phoneme_Position', 'Phoneme_Type', or 'Phoneme_Group'.")
    numrun = tsne_df['#Run'].iloc[0]
    method = tsne_df['Method'].iloc[0]
    patient = tsne_df['Patient'].iloc[0]
    fig = px.scatter(tsne_df, x='tsne-1', y=-tsne_df['tsne-2'], color=huere, opacity=0.7,
                     color_discrete_sequence = plette, 
                     category_orders=catorder)
    fig.update_layout(
        title= f"TSNE Colored on: {huere} - Silhouette Score: {silscore:.3f} - Patient: {patient} - Method: {method} #Run: {numrun}", 
        legend_title_text=huere)
    #fig.update_traces(marker=dict(size=40))
    fig.update_xaxes(showticklabels=False, showline=True, linecolor='black')
    fig.update_yaxes(showticklabels=False, showline=True, linecolor='black')
    fig.write_image(rf"Figures\Figure 3\TSNE\TSNE on {huere} Silscore {silscore:.2f} {patient} {method} {numrun}.svg", format='svg',scale=3)
    #fig.show()
    return fig


