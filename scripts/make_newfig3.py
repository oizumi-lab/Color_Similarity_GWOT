"""
Make Fig. 3

Genji Kawakita
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *
import sqlite3

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
from GW_methods.src.align_representations import Representation, VisualizationConfig, AlignRepresentations, OptimizationConfig
from src.utils.plot_utils import plot_MDS_embedding

# %% Load data
data_list = ["neutyp", "atyp", "n-a"]# "neutyp" : n-n, "atyp" : a-a
N_groups_list = [2, 2, 2] # number of groups for each data type4, 
Z_list = [128]# 20, 60, # number of participants
N_sample = 1 # number of sampling
N_trials = 75
idx = 2
data = data_list[idx]
N_groups = N_groups_list[idx]
Z = Z_list[0]

# load color codes
old_color_order = list(np.load('../data/hex_code/original_color_order.npy'))
new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
# get the reordering indices
reorder_idxs = get_reorder_idxs(old_color_order,new_color_order)


embeddings_pairs_list = np.load(f"../results/embeddings_pairs_list_{data}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}.npy")
sim_mat_list = []
for i, embeddings_pair in enumerate(embeddings_pairs_list):
    
    representations = []
    for j, embedding in enumerate(embeddings_pair):
        representation = Representation(name=f"{data}-{j+1}", embedding=embedding[reorder_idxs, :], metric="euclidean")
        sim_mat_list.append(representation.sim_mat)

# %% Fig 3.a (N1 & N2 RDM) | Fig 4.a (A1 & A2 RDM)
# load all data

# plot and save figs
save_dir = "../results/figs"
#cmap = cmap_discretize("gist_gray",8)
#cmap = "gist_gray"
for i in range(N_groups):
    plt.figure(figsize=(20,20))
    sns.heatmap(sim_mat_list[i],cbar=False,cmap="gist_gray",vmin=0, vmax=8)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rdm_{data}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}_group{i+1}.png',dpi=300)
    plt.show()
# %% Fig 3.b/4.b(Optimal transportation plan)
gw_path = f"../results/gw_alignment/color_{data}_Z={Z}_Ntrials={N_trials}_seed0/color_{data}_Z={Z}_Ntrials={N_trials}_seed0_{data}-1_vs_{data}-2/random"
# load .db file
db_file = os.path.join(gw_path,f"color_{data}_Z={Z}_Ntrials={N_trials}_seed0_{data}-1_vs_{data}-2_random.db")
# connect to db
conn = sqlite3.connect(db_file)
# display tables
cursor = conn.cursor()
cursor.execute("SELECT * FROM trial_values;")
studies = cursor.fetchall()
print(studies[0])
print(studies[0][3])
# find the best OT (minimum GW distance studies[i][3])
min_gwd= 100
best_OT_idx = 0
for i in range(len(studies)):
    if studies[i][3] < min_gwd:
        min_gwd = studies[i][3]
        best_OT_idx = i + 1

# load the best OT
best_OT = np.load(f"{gw_path}/data/gw_{best_OT_idx}.npy")
max_val = 1/93
fig,ax = plt.subplots(1,1,figsize=(20,20))
a = sns.heatmap(best_OT,ax=ax,cbar=False,vmax=max_val) 
plt.xticks([])
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'{save_dir}/best_ot_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}_{data}-1_vs_{data}-2.png',dpi=300)

#%% Correlation plots

# %% Fig 3.c/4.c eps v.s. GW distance


# %% Individual embeddings
marker = "o"
view_init = (15,150)
data = "atyp"
embeddings_pairs_list = np.load(f"../results/embeddings_pairs_list_{data}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}.npy")
for i, embeddings_pair in enumerate(embeddings_pairs_list):
    for j, embedding in enumerate(embeddings_pair):
        representation = Representation(name=f"{data}-{j+1}", embedding=embedding[reorder_idxs, :], metric="euclidean")
        embeddings = representation.embedding
        save_name = f"individual_{data}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}_group{i+1}_embedding{j+1}.png"
        # visualize in 3D
        plot_MDS_embedding(embeddings[:,:3],marker,new_color_order,alpha=1,s=300,\
                   fig_size=(15,15),save=True,fig_dir=save_dir,save_name=save_name,view_init=view_init)

# %% Fig 3.d (Aligned embeddings)
# apply procrstes to align Group 2-5 embeddings (Y_i) to Group 1 (X)
aligned_embedding_list = []
for i in range(1,n_groups):
    X = embedding_list[0]
    Y = embedding_list[i]
    OT = best_OTs[i-1]
    _, aligned_embedding = procrustes_alignment(X,Y,OT)
    aligned_embedding_list.append(aligned_embedding)


# plot aligned embeddings
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection="3d")
for grp_idx in range(n_groups):
    if grp_idx == 0:
        ax.scatter(xs=embedding_list[grp_idx][:,0],ys=embedding_list[grp_idx][:,1],zs=embedding_list[grp_idx][:,2],\
            marker=markers_list[grp_idx],color=new_color_order,alpha=1,s=300,label=f"Group {grp_idx+1}")
    else:
        ax.scatter(xs=aligned_embedding_list[grp_idx-1][:,0],ys=aligned_embedding_list[grp_idx-1][:,1],zs=aligned_embedding_list[grp_idx-1][:,2],\
            marker=markers_list[grp_idx],color=new_color_order,alpha=1,s=300,label=f"Group {grp_idx+1}")
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_zaxis().set_visible(False)
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.set_facecolor('white')
ax.view_init(elev=15, azim=150)
ax.legend(fontsize=24,loc=(0.8,0.6))
leg = ax.get_legend()
for i in range(n_groups):
    leg.legendHandles[i].set_color('k')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,"fig3d_aligned_embeddings.png"))
plt.show()
    
