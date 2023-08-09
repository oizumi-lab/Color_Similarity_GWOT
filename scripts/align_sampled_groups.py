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

#from src.preprocess_utils import *
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

#%%
data_list = ["neutyp", "atyp" , "n-a"]#
Z_list = [100]# 20, 60, 
N_sample = 1 # number of sampling
N_trials = 76

# load color codes
old_color_order = list(np.load('../data/hex_code/original_color_order.npy'))
new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
# get the reordering indices
reorder_idxs = get_reorder_idxs(old_color_order,new_color_order)

#%%
for Z in Z_list:
    for data in data_list:  
        
        embeddings_pairs_list = np.load(f"../results/embeddings_pairs_list_{data}_Z={Z}_Ntrials={N_trials}_Nsample={N_sample}.npy")
        
        ### set accuracy dataframe
        top_k_list = [1, 3, 5]
        top_k_accuracy = pd.DataFrame()
        top_k_accuracy["top_n"] = top_k_list
        
        k_nearest_matching_rate = pd.DataFrame()
        k_nearest_matching_rate["top_n"] = top_k_list
        
        ### alignment 
        for i, embeddings_pair in enumerate(embeddings_pairs_list):
            group1 = Representation(name=f"Group1 seed={i}", embedding=embeddings_pair[0][reorder_idxs, :], metric="euclidean")
            group2 = Representation(name=f"Group2 seed={i}", embedding=embeddings_pair[1][reorder_idxs, :], metric="euclidean")

            representation_list = [group1, group2]

            opt_config = OptimizationConfig(data_name=f"color {data} Z={Z} Ntrials={N_trials}", 
                                    init_mat_plan="random",
                                    db_params={"drivername": "sqlite"},
                                    num_trial=100,
                                    n_iter=2, 
                                    max_iter=200,
                                    sampler_name="tpe", 
                                    eps_list=[0.02, 0.2],
                                    eps_log=True,
                                    )
            
            alignment = AlignRepresentations(config=opt_config, 
                                    representations_list=representation_list,
                                    metric="euclidean",
                                    )
            
            vis_config = VisualizationConfig(
                figsize=(8, 6), 
                title_size = 15, 
                cmap = "rocket",
                cbar_ticks_size=10,
                )
            
            alignment.show_sim_mat(
                visualization_config=vis_config, 
                show_distribution=False,
                fig_dir=f"../results/figs/{data}/"
                )
            alignment.RSA_get_corr()
            
            alignment.gw_alignment(results_dir="../results/gw alignment/",
                        compute_OT=False,
                        delete_results=False,
                        visualization_config=vis_config,
                        fig_dir="../results/figs/"
                        )
            
            vis_emb = VisualizationConfig(
                figsize=(8, 8), 
                legend_size=12,
                marker_size=60,
                color_labels=new_color_order,
                markers_list=["o", "^"]
                )
            alignment.visualize_embedding(dim=3, visualization_config=vis_emb, fig_dir=f"../results/figs/{data}", fig_name=f"{data}_Aligned_embedings.svg")
            
            ## Calculate the accuracy of the optimized OT matrix
            alignment.calc_accuracy(top_k_list=top_k_list, eval_type="ot_plan")

            ## Calculate the accuracy based on k-nearest neighbors
            alignment.calc_accuracy(top_k_list=top_k_list, eval_type="k_nearest")

            top_k_accuracy = pd.merge(top_k_accuracy, alignment.top_k_accuracy, on="top_n")
            k_nearest_matching_rate = pd.merge(k_nearest_matching_rate, alignment.k_nearest_matching_rate, on="top_n")
            
        top_k_accuracy.to_csv(f"../results/top_k_accuracy_{data}_Z={Z}_Nsample={N_sample}.csv")
        k_nearest_matching_rate.to_csv(f"../results/k_nearest_matching_rate_{data}_Z={Z}_Nsample={N_sample}.csv")
# %%

