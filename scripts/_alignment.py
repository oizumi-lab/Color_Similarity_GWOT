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
from GW_methods.src.align_representations import Representation, Visualization_Config, Align_Representations, Optimization_Config

#%%

N_groups = 2
data = "atyp" #"neutyp" or "atyp"
barycenter = False

'''
Unsupervised alignment
'''

rearranged_color_embeddings_list = np.load(f"../results/rearranged_embeddings_list_{data}_Ngroup={N_groups}.npy")
representation_list = []
for i in range(N_groups):
    representation = Representation(name=f"Group{i+1}_{data}", embedding=rearranged_color_embeddings_list[i], metric="euclidean")
    representation_list.append(representation)


opt_config = Optimization_Config(data_name=f"color_Ngroup={N_groups}", 
                                    init_plans_list=["random"],
                                    num_trial=10,
                                    n_iter=10, 
                                    max_iter=200,
                                    sampler_name="tpe", 
                                    eps_list=[0.02, 0.2],
                                    eps_log=True,
                                    )

alignment = Align_Representations(config=opt_config, 
                                    representations_list=representation_list,
                                    metric="euclidean",
                                    )

vis_config = Visualization_Config()
alignment.show_sim_mat(returned="figure",
                        visualization_config=vis_config,
                        show_distribution=False
                        )
alignment.RSA_get_corr()

# GW alignment
if not barycenter:
    alignment.gw_alignment(results_dir="../results/gw alignment/",
                        load_OT=False,
                        returned="figure",
                        visualization_config=vis_config,
                        show_log=False,
                        fig_dir="../results/figs/"
                        )
if barycenter:
    alignment.barycenter_alignment(pivot=0, 
                                    n_iter=30, 
                                    results_dir="../results/gw alignment/",
                                    load_OT=True,
                                    returned="figure",
                                    visualization_config=vis_config,
                                    show_log=False,
                                    fig_dir="../results/figs/"
                                    )

#%%
barycenter_str = "_barycenter" if barycenter else ""
## Calculate the accuracy of the optimized OT matrix
alignment.calc_accuracy(top_k_list = [1, 3, 5], eval_type = "ot_plan", barycenter=barycenter)
alignment.plot_accuracy(eval_type = "ot_plan", scatter = True, fig_dir="../results/figs", fig_name = f"top_k_accuracy_{data}_Ngroup{N_groups}{barycenter_str}")

## Calculate the accuracy based on k-nearest neighbors
alignment.calc_accuracy(top_k_list = [1, 3, 5], eval_type = "k_nearest", barycenter=barycenter)
alignment.plot_accuracy(eval_type = "k_nearest", scatter = True, fig_dir="../results/figs", fig_name = f"k_nearest_rate_{data}_Ngroup{N_groups}{barycenter_str}")

alignment.top_k_accuracy.to_csv(f"../results/top_k_accuracy_{data}_Ngroup={N_groups}{barycenter_str}.csv")
alignment.k_nearest_matching_rate.to_csv(f"../results/k_nearest_rate_{data}_Ngroup={N_groups}{barycenter_str}.csv")

#%%
#Plot embedding
new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
vis_config_emb = Visualization_Config(color_labels=new_color_order,
                                        marker_size=10)

pivot = "barycenter" if barycenter else 0
alignment.visualize_embedding(dim=3,
                                pivot=pivot, 
                                returned="figure",
                                visualization_config=vis_config_emb,
                                fig_dir="../results/figs",
                                fig_name = f"Aligned_embeddings_{data}_Ngroup{N_groups}{barycenter_str}"
                                )

