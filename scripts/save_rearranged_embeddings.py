#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import itertools
import scipy.spatial.distance as dist

#from src.preprocess_utils import *
from src.utils.utils import get_reorder_idxs

#%%

data_list = ["neutyp", "atyp", "n-a"]# "neutyp" : n-n, "atyp" : a-a, 
N_groups_list = [2, 2, 2] # number of groups for each data type4, , 2, 2
Z_list = [128]# 20, 60, # number of participants, 128, 16, 32, 64, 
N_sample = 20 # number of sampling
N_trials = 75
emb_dim = 20

# load color codes
old_color_order = list(np.load('../data/hex_code/original_color_order.npy'))
new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
# get the reordering indices
reorder_idxs = get_reorder_idxs(old_color_order,new_color_order)

#%%
for Z in Z_list:
    for data, N_groups in zip(data_list, N_groups_list):  
        embeddings_pairs_list = np.load(f"../results/embeddings_pairs_list_{data}_emb={emb_dim}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}{'_independent' if data=='n-a' else ''}.npy")
        
        for i in range(2):
            embedding = embeddings_pairs_list[0][i]
            embedding=embedding[reorder_idxs, :]
            
            np.save(f"../results/rearranged_embeddings_{data}_emb={emb_dim}_Z={Z}_group{i}_Ntrials={N_trials}.npy", embedding)
            
            # save the matrix of pairwise distances
            dist_matrix = dist.cdist(embedding, embedding, metric='euclidean')
            np.save(f"../results/rearranged_RDM_{data}_emb={emb_dim}_Z={Z}_group{i}_Ntrials={N_trials}.npy", dist_matrix)
        
# %%
