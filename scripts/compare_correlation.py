
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *
import sqlite3
from scipy.stats import spearmanr

from src.utils.utils import get_reorder_idxs
#from src.plot_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

import ot
from sklearn.metrics import pairwise_distances

from src.embedding_model import EmbeddingModel, ModelTraining
from GWTune.src.align_representations import Representation, VisualizationConfig, AlignRepresentations, OptimizationConfig
from src.utils.plot_utils import plot_MDS_embedding

# %% Load data
data_list = ["neutyp", "atyp", "n-a"]# "neutyp" : n-n, "atyp" : a-a
N_groups_list = [2, 2, 2] # number of groups for each data type4, 
Z_list = [128]# 20, 60, # number of participants  16,32,64,
N_sample = 20  # number of sampling
N_trials = 75
emb_dim = 20
# load color codes
old_color_order = list(np.load('../data/hex_code/original_color_order.npy'))
new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
# get the reordering indices
reorder_idxs = get_reorder_idxs(old_color_order,new_color_order)

all_dict = {}
for idx in range(len(data_list)):
    data = data_list[idx]
    N_groups = N_groups_list[idx]
    for Z in Z_list:

        embeddings_pairs_list = np.load(f"../results/embeddings_pairs_list_{data}_emb={emb_dim}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}{'_independent' if data=='n-a' else ''}.npy")
        sim_mat_list = []
        corr_list = []
        for i, embeddings_pair in enumerate(embeddings_pairs_list):
            #print(i)
            #representations = []
            #for j, embedding in enumerate(embeddings_pair):
            embedding1 = embeddings_pair[0]
            embedding2 = embeddings_pair[1]
            
            representation1 = Representation(name=f"{data}-{1}", embedding=embedding1[reorder_idxs, :], metric="euclidean")
            representation2 = Representation(name=f"{data}-{2}", embedding=embedding2[reorder_idxs, :], metric="euclidean")
            
            sim_mat1 = representation1.sim_mat
            sim_mat2 = representation2.sim_mat
            
            upper_tri_1 = sim_mat1[np.triu_indices(sim_mat1.shape[0], k=1)]
            upper_tri_2 = sim_mat2[np.triu_indices(sim_mat2.shape[0], k=1)]
            corr, _ = spearmanr(upper_tri_1, upper_tri_2)
            corr_list.append(corr)
        all_dict[f"{data}_Z={Z}"] = corr_list
# %%
# Convert dictionary to DataFrame
df_list = []
for key, value in all_dict.items():
    data, Z = key.split('_Z=')
    for corr in value:
        df_list.append([data, int(Z), corr])

df = pd.DataFrame(df_list, columns=['data_type', 'Z', 'correlation'])
name_mapping = {
    'neutyp': 'N v.s. N',
    'atyp': 'A v.s. A',
    'n-a': 'N v.s. A'
}
df["data_type"] = df["data_type"].replace(name_mapping)
# Create swarmplot
labels = [x * N_trials for x in Z_list]

palette = sns.color_palette("bright", n_colors=len(labels))

plt.figure(figsize=(6, 6))

sns.swarmplot(data=df, x='Z', y='correlation', hue='data_type', dodge=True, palette='bright', size=3)
labels = ["1200\n(Z=16)", "2400\n(Z=32)", "4800\n(Z=64)", "9600\n(Z=128)"]
# Customize ticks and labels
plt.xticks(ticks=[0, 1, 2, 3], labels=labels)
plt.xlabel('Color combinations', size=15)
plt.ylabel(f'Spearmann Correlations', size=15)
plt.legend()

#plt.xticks(ticks=Z_list, labels=labels, size=25)
plt.xticks(size=15)
plt.yticks(size=15)

# 凡例を表示する
plt.legend(fontsize=15, loc='upper left')
plt.tight_layout()
plt.show()
# %%
