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

def generate_random_grouping(N_participant, N_groups, seed = 0):
    random.seed(seed)
    
    participants = list(range(N_participant))
    random.shuffle(participants)  # リストをランダムにシャッフルする

    group_size = N_participant // N_groups  # グループごとの理想的なサイズ
    remainder = N_participant % N_groups  # 割り切れなかった場合の余り

    groups = []
    start = 0
    for i in range(N_groups):
        group_end = start + group_size + (1 if i < remainder else 0)  # グループの終了位置
        groups.append(participants[start:group_end])
        start = group_end

    return groups

#%%
class MakeDataset:
    def __init__(self, participants_list, data_dir = "../data/color_neurotypical/numpy_data/"):
        self.participants_list = participants_list 
        self.data_dir = data_dir
    
    def get_response_array(self):
        # Create empty lists to store X and y
        X_np = []
        y_np = []
        data_list = []
        
        # Loop over the file names
        for i in self.participants_list:
            # Load the data file
            filepath = f"participant_{i}.npy"
            data = np.load(self.data_dir + filepath, allow_pickle=True)
            data = data.astype(np.float64)
            data_list.append(data)

            # Get lower triangle indices where data is not NaN
            lower_triangle_indices = np.tril_indices(data.shape[0], -1)  # -1 excludes the diagonal
            values = data[lower_triangle_indices]
            non_nan_indices = np.where(~np.isnan(values))

            # Get final indices and values
            final_indices = (lower_triangle_indices[0][non_nan_indices], lower_triangle_indices[1][non_nan_indices])
            final_values = values[non_nan_indices]

            # Append the indices (X) and values (y) to the lists
            X_np.extend(list(zip(*final_indices)))  # Zip the indices to get (row, col) pairs
            y_np.extend(final_values)

        # Convert lists to numpy arrays
        X_np = np.array(X_np)
        y_np = np.array(y_np)

        return X_np, y_np
    
    def __call__(self):
        X_np, y_np = self.get_response_array()

        #Convert numpy arrays to PyTorch tensors
        X_tensor = torch.LongTensor(X_np)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

        # Create a TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        return dataset
    
    
class MainTraining():
    def __init__(self, dataset, test_size, batch_size, device) -> None:
        self.dataset = dataset
        self.test_size = test_size
        self.batch_size = batch_size
        self.device = device
        
        pass

    def train_test_split_dataset(self):
        train_dataset, valid_dataset = train_test_split(self.dataset, test_size = self.test_size, shuffle = False, random_state = 42)

        return train_dataset, valid_dataset
    
    def make_dataloader(self):
        train_dataset, valid_dataset = self.train_test_split_dataset()
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle = True)
        valid_dataloader = DataLoader(valid_dataset, self.batch_size, shuffle = False)
        
        return train_dataloader, valid_dataloader
    
    def main_compute(self, loss_fn, emb_dim, object_num, n_epoch, lr, distance_metric = "euclidean", lamb = None):
        train_dataloader, valid_dataloader = self.make_dataloader()
        
        model = EmbeddingModel(emb_dim = emb_dim, object_num = object_num).to(self.device)
        
        model_training = ModelTraining(self.device, 
                                       model = model, 
                                       train_dataloader = train_dataloader, 
                                       valid_dataloader = valid_dataloader, 
                                       similarity = "pairwise", 
                                       distance_metric = distance_metric)
        
        test_loss = model_training.main_compute(loss_fn = loss_fn, 
                                                lr = lr, 
                                                num_epochs = n_epoch, 
                                                lamb = lamb, 
                                                show_log= True)
        
        weights = model.state_dict()["Embedding.weight"].to('cpu').detach().numpy().copy()
        
        return weights
    

#%%
if __name__ == "__main__":
    ### Set parameters
    N_participant = 426 # max : 426
    N_groups = 8
    data_dir = "../data/color_neurotypical/numpy_data/"

    participants_list = generate_random_grouping(N_participant=N_participant, 
                                                 N_groups=N_groups, 
                                                 seed=0)
    
    # device 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train the model
    emb_dim = 5
    color_num = 93
    
    # Define the optimizer (e.g., Adam)
    lr = 0.01
    num_epochs = 100
    batch_size = 100

    loss_fn = nn.MSELoss()

    # load color codes
    old_color_order = list(np.load('../data/hex_code/original_color_order.npy'))
    new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
    # get the reordering indices
    reorder_idxs = get_reorder_idxs(old_color_order,new_color_order)
    #%%
    ### Main computation
    rearranged_color_embeddings_list = []
    for i in range(N_groups):
        dataset = MakeDataset(participants_list=participants_list[i], data_dir=data_dir)

        main_training = MainTraining(dataset = dataset(), 
                                     test_size = 0.2, 
                                     batch_size = batch_size, 
                                     device = device)

        color_embeddings = main_training.main_compute(loss_fn = loss_fn, 
                                   emb_dim = emb_dim, 
                                   object_num = color_num, 
                                   n_epoch = num_epochs, 
                                   lr = lr, 
                                   distance_metric = "euclidean")

        rearranged_color_embeddings_list.append(color_embeddings[reorder_idxs, :])
    np.save(f"../results/rearranged_embeddings_list_Ngroup={N_groups}.npy", rearranged_color_embeddings_list)
    #%%
    rearranged_color_embeddings_list = np.load(f"../results/rearranged_embeddings_list_Ngroup={N_groups}.npy")
    representation_list = []
    for i in range(N_groups):
        representation = Representation(name=f"Group{i+1}_neutyp", embedding=rearranged_color_embeddings_list[i], metric="euclidean")
    
        #vis_config = Visualization_Config()
        #representation.show_sim_mat(returned="figure", visualization_config=vis_config)
#
        #vis_config_emb = Visualization_Config(color_labels=new_color_order)
        #representation.show_embedding(dim=3, visualization_config=vis_config_emb)
        
        representation_list.append(representation)
        
    opt_config = Optimization_Config(data_name="color", 
                                     init_plans_list=["random"],
                                     num_trial=4,
                                     n_iter=3, 
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
    alignment.show_sim_mat(returned="row_data",
                           visualization_config=vis_config,
                           )
    alignment.RSA_get_corr()
    
    # GW alignment
    alignment.gw_alignment(results_dir="../results/gw alignment/",
                           load_OT=True,
                           returned="row_data",
                           visualization_config=vis_config,
                           show_log=False,
                           fig_dir="../results/figs/",
                           )
    
    #%%
    ## Calculate the accuracy of the optimized OT matrix
    alignment.calc_accuracy(top_k_list = [1, 3, 5], eval_type = "ot_plan")
    alignment.plot_accuracy(eval_type = "ot_plan", scatter = True, fig_dir="../results/figs", fig_name = f"matching_rate_neutyp_Ngroup{N_groups}")
    
    #%%
    #Plot embedding
    vis_config_emb = Visualization_Config(color_labels=new_color_order,
                                          marker_size=10)
    alignment.visualize_embedding(dim = 3,
                                  returned="figure",
                                  visualization_config=vis_config_emb,
                                  fig_dir="../results/figs",
                                  fig_name = f"Aligned_embeddings_neutyp_Ngroup{N_groups}"
                                  )


## %% check the prediction of the model
#data_sim_list = []
#model_sim_list = []
#for i, pair in enumerate(X_np):
#    data_sim = y_np[i]
#    data_sim_list.append(data_sim)
#    
#    x1 = color_embeddings[pair[0],:]
#    x2 = color_embeddings[pair[1],:]
#    model_sim = np.linalg.norm(x1-x2)
#    model_sim_list.append(model_sim)
#    
#df = pd.DataFrame(data={'data': data_sim_list, 'model': model_sim_list})
# %%
