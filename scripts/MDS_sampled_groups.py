#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import random
import numpy as np
import torch
import torch.nn as nn

from color_embedding_MDS import MakeDataset, MainTraining, KFoldCV
from src.utils.utils import get_reorder_idxs

#%%
def sample_participants(N, Z, seed):
    random.seed(seed)
    participants_list = random.sample(range(N), Z)
    return participants_list

def split_lists(list, n_groups): 
    N = len(list) // n_groups
    result = []
    for i in range(n_groups):
        list_i = list[N*i : N*(i+1)]
        result.append(list_i)
    return result

#%%
if __name__ == "__main__": 
    ### set parameters
    # device 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train the model
    emb_dim = 5
    color_num = 93

    # Define the optimizer (e.g., Adam)
    lr = 0.01
    num_epochs = 100
    early_stopping = True
    batch_size = 100

    loss_fn = nn.MSELoss(reduction="sum")
    
    ### cv params
    n_splits = 5
    lamb_range = [1e-2, 1e+3]
    cv_n_trial = 1
                    
                    
    data_list = ["neutyp"]#"atyp", "atyp"
    N_trials = 75
    
    Z_list = [128] # number of participants per group
    N_groups = 2
    
    N_sample = 1 # number of sampling
    seed_list = range(N_sample)
    
    #%%
    for data in data_list:
        for Z in Z_list:
            ### load data
            if data == "neutyp":
                N_participant = 426
                data_dir = "../data/color_neurotypical/numpy_data/"
        
            elif data == "atyp":
                N_participant = 257
                data_dir = "../data/color_atypical/numpy_data/"
            
            # load color codes
            old_color_order = list(np.load('../data/hex_code/original_color_order.npy'))
            new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
            # get the reordering indices
            reorder_idxs = get_reorder_idxs(old_color_order,new_color_order)
            
            
            ### set participants list
            group_pairs_list = []
            for seed in seed_list:
                sampled_participants = sample_participants(N=N_participant, Z=Z*N_groups, seed=seed)
                group_pair = split_lists(sampled_participants, N_groups)
                group_pairs_list.append(group_pair)
                
            
            ### estimate embeddings
            embeddings_pairs_list = []
            for i, group_pair in enumerate(group_pairs_list):
                
                embeddings_pair = []
                for j, participants_list in enumerate(group_pair):
                    dataset = MakeDataset(participants_list=participants_list, data_dir=data_dir)
                    
                    study_name = f"{data} Z={Z} seed={i} group{j+1}"
                    
                    ### cross validation
                    cv = KFoldCV(dataset=dataset(),
                                 n_splits=n_splits,
                                 search_space=lamb_range,
                                 study_name=study_name,
                                 results_dir=f"../results/cross validation/{data}/",
                                 batch_size=batch_size,
                                 device=device,
                                 loss_fn=loss_fn,
                                 emb_dim=emb_dim,
                                 object_num=color_num,
                                 n_epoch=num_epochs,
                                 lr=lr,
                                 early_stopping=early_stopping,
                                 distance_metric="euclidean")

                    cv.optimize(n_trials=cv_n_trial)
                    lamb = cv.get_best_lamb(show_log=True)
                    
                    ### main
                    main_training = MainTraining(dataset = dataset(N_trials=N_trials), 
                                             test_size = 1/n_splits, 
                                             batch_size = batch_size, 
                                             device = device)
                    
                    color_embeddings, loss = main_training.main_compute(loss_fn = loss_fn, 
                                           emb_dim = emb_dim, 
                                           object_num = color_num, 
                                           n_epoch = num_epochs, 
                                           lr = lr, 
                                           early_stopping=early_stopping,
                                           distance_metric = "euclidean", 
                                           lamb=lamb)
                    
                    embeddings_pair.append(color_embeddings)
        
                embeddings_pairs_list.append(embeddings_pair)
                
            np.save(f"../results/embeddings_pairs_list_{data}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}.npy", embeddings_pairs_list)
            
    #%%
    ### set n-a embedding pairs
    N_groups_N = 4
    N_groups_A = 2
    for Z in Z_list:
        embeddings_pairs_list_N = np.load(f"../results/embeddings_pairs_list_neutyp_Z={Z}_Ngroups={N_groups_N}_Ntrials={N_trials}_Nsample={N_sample}.npy")
        embeddings_pairs_list_A = np.load(f"../results/embeddings_pairs_list_atyp_Z={Z}_Ngroups={N_groups_A}_Ntrials={N_trials}_Nsample={N_sample}.npy")

        embeddings_pairs_list = []
        for i in range(N_sample):
            embeddings_pair = [embeddings_pairs_list_N[i][0], embeddings_pairs_list_A[i][0]]
            embeddings_pairs_list.append(embeddings_pair)

        np.save(f"../results/embeddings_pairs_list_n-a_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}.npy", embeddings_pairs_list)
# %%
