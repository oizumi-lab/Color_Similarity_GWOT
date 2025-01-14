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
from src.utils.plot_utils import gif_animation
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
compute_OT = False

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
        
        ### set accuracy dataframe
        top_k_list = [1, 3, 5]
        top_k_accuracy = pd.DataFrame()
        top_k_accuracy["top_n"] = top_k_list
        
        k_nearest_matching_rate = pd.DataFrame()
        k_nearest_matching_rate["top_n"] = top_k_list
        
        df_rsa = pd.DataFrame()
        
        ### alignment 
        for i, embeddings_pair in enumerate(embeddings_pairs_list):
            
            vis_emb = VisualizationConfig(
                figsize=(8, 8), 
                fig_ext="svg",
                legend_size=12,
                marker_size=60,
                color_labels=new_color_order,
                font = "Arial",
                title_size = 50
                )
            
            if data == "neutyp":
                name_list = ["Group T1", "Group T2"]
            elif data == "atyp":
                name_list = ["Group A1", "Group A2"]
            elif data == "n-a":
                name_list = ["Group T1", "Group A1"]
                
            os.makedirs(f"../results/figs/{data}/Z={Z}/seed{i}/", exist_ok=True)
            # save
            save_dir = f"../results/{data}/Z={Z}/seed{i}/"
            os.makedirs(save_dir, exist_ok=True)
            
            representations = []
            for j, embedding in enumerate(embeddings_pair):
                representation = Representation(name=f"{data}-{j+1}", embedding=embedding[reorder_idxs, :], metric="euclidean")
                representations.append(representation)
                representation.show_embedding(dim=3, visualization_config=vis_emb, legend=None, title=None, fig_dir=f"../results/figs/{data}/Z={Z}/seed{i}/", fig_name=f"{data}_seed{i}_{j}_embedings.png")
                # save
                np.save(os.path.join(save_dir, f"embedding_{data}_{j}.npy"), embedding)
            
            opt_config = OptimizationConfig(
                                    init_mat_plan="random",
                                    db_params={"drivername": "sqlite"},
                                    num_trial=200,
                                    n_iter=1, 
                                    max_iter=200,
                                    sampler_name="tpe", 
                                    eps_list=[0.02, 0.2],
                                    eps_log=True,
                                    )
            
            alignment = AlignRepresentations(config=opt_config, 
                                    representations_list=representations,
                                    metric="euclidean",
                                    main_results_dir="../results/gw_alignment/",
                                    data_name=f"color_{data}_Z={Z}_Ntrials={N_trials}_seed{i}{'_independent' if data=='n-a' else ''}", 
                                    )
            
            vis_config = VisualizationConfig(
                figsize=(8, 6), 
                title_size = 15, 
                cmap = "gray",
                cbar_ticks_size=30,
                font = "Arial",
                cbar_label="Dissimilarity",
                cbar_label_size=40,
                color_labels = new_color_order,
                color_label_width = 5,
                )
            
            alignment.show_sim_mat(
                visualization_config=vis_config, 
                show_distribution=False,
                fig_dir=f"../results/figs/{data}/Z={Z}/seed{i}/"
                )
            alignment.RSA_get_corr(metric='pearson')
            rsa_corr = pd.DataFrame([alignment.RSA_corr], index=['correlation'])
            df_rsa = pd.concat([df_rsa, rsa_corr], axis=1)
            
            vis_config_OT = VisualizationConfig(
                figsize=(8, 6), 
                title_size = 15, 
                cmap = "rocket",
                cbar_ticks_size=30,
                font = "Arial",
                cbar_label="Probability",
                cbar_label_size=40,
                color_labels = new_color_order,
                color_label_width = 5,
                xlabel=f"93 colors of {name_list[0]}",
                xlabel_size = 40,
                ylabel=f"93 colors of {name_list[1]}",
                ylabel_size = 40,
                )
            
            alignment.gw_alignment(
                        compute_OT=compute_OT,
                        delete_results=False,
                        visualization_config=vis_config_OT,
                        fig_dir=f"../results/figs/{data}/Z={Z}/seed{i}/"
                        )
            # save the best ot
            OT = alignment.gw_alignment(
                        compute_OT=compute_OT,
                        delete_results=False,
                        return_data=True,
                        return_figure=False,
                        visualization_config=vis_config_OT,
                        )
            np.save(os.path.join(save_dir, f"OT_{data}.npy"), OT)
            
            vis_config_log = VisualizationConfig(
                figsize=(8, 6), 
                title_size = 15, 
                cbar_ticks_size=30,
                font = "Arial",
                fig_ext="svg",
                xlabel_size=35,
                xticks_size=30,
                xticks_rotation=0,
                ylabel_size=35,
                yticks_size=30,
                cbar_label_size=30
                )
            
            alignment.show_optimization_log(visualization_config=vis_config_log, fig_dir=f"../results/figs/{data}/Z={Z}/seed{i}/")
            
            vis_emb = VisualizationConfig(
                figsize=(8, 8), 
                legend_size=30,
                marker_size=60,
                color_labels=new_color_order,
                font = "Arial",
                fig_ext = "svg",
                )
            
                
            embedding_aligned = alignment.visualize_embedding(dim=3, visualization_config=vis_emb, fig_dir=f"../results/figs/{data}/Z={Z}/seed{i}/", fig_name=f"{data}_Aligned_embedings", name_list=name_list)
            
            # get embedding
            embedding_list = alignment.visualize_embedding(dim=3, returned="row_data")
            
            np.save(os.path.join(save_dir, f"embedding_aligned.npy"), embedding_aligned)
            
            # make animation
            #gif_animation(embedding_list=embedding_aligned, markers_list=['o', 'x'], colors=new_color_order, fig_size=(15,12), save_anim=True, save_path=f"../results/figs/{data}/Z={Z}/seed{i}/alignment.gif")
            #gif_animation(embedding_list=embedding_aligned, markers_list=['o', 'x'], colors=new_color_order, fig_size=(15,12), save_anim=True, save_path=f"../results/figs/{data}/Z={Z}/seed{i}/alignment.mp4")
            
            ## Calculate the accuracy of the optimized OT matrix
            alignment.calc_accuracy(top_k_list=top_k_list, eval_type="ot_plan")

            ## Calculate the accuracy based on k-nearest neighbors
            alignment.calc_accuracy(top_k_list=top_k_list, eval_type="k_nearest")

            alignment.plot_accuracy(eval_type="ot_plan", fig_dir=f"../results/figs/{data}/Z={Z}/seed{i}/", fig_name="accuracy_ot_plan.png")
            alignment.plot_accuracy(eval_type="k_nearest", fig_dir=f"../results/figs/{data}/Z={Z}/seed{i}/", fig_name="accuracy_k_nearest.png")
            
            top_k_accuracy = pd.merge(top_k_accuracy, alignment.top_k_accuracy, on="top_n")
            k_nearest_matching_rate = pd.merge(k_nearest_matching_rate, alignment.k_nearest_matching_rate, on="top_n")
            
        top_k_accuracy.to_csv(f"../results/top_k_accuracy_{data}_Z={Z}_Nsample={N_sample}.csv")
        k_nearest_matching_rate.to_csv(f"../results/k_nearest_matching_rate_{data}_Z={Z}_Nsample={N_sample}.csv")
        df_rsa.to_csv(f"../results/rsa_corr_{data}_Z={Z}_Nsample={N_sample}.csv")
# %%
print("finished")

# %%
