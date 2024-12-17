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
import optuna
from optuna.storages import RDBStorage

#from src.preprocess_utils import *
from src.utils.utils import get_reorder_idxs
#from src.plot_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.model_selection import train_test_split, KFold

from joblib import Parallel, delayed

import ot
from sklearn.metrics import pairwise_distances

from src.embedding_model import EmbeddingModel, ModelTraining
from GW_methods.src.align_representations import Representation, VisualizationConfig, AlignRepresentations, OptimizationConfig

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
    
    def get_response_array(self, N_trials=None):
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

            X = list(zip(*final_indices)) # Zip the indices to get (row, col) pairs
            y = list(final_values)
            
            if N_trials is not None:
                X = self.sample_trials(X, N_trials=N_trials)
                y = self.sample_trials(y, N_trials=N_trials)
                
            # Append the indices (X) and values (y) to the lists
            X_np.extend(X) 
            y_np.extend(y)


        # Convert lists to numpy arrays
        X_np = np.array(X_np)
        y_np = np.array(y_np)

        return X_np, y_np
    
    def sample_trials(self, list, N_trials, seed=0):
        random.seed(seed)
        new_list = random.sample(list, N_trials)

        return new_list
        
    def __call__(self, N_trials=None):
        X_np, y_np = self.get_response_array(N_trials=N_trials)
    
        #Convert numpy arrays to PyTorch tensors
        X_tensor = torch.LongTensor(X_np)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

        # Create a TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        return dataset
    
    
class MainTraining():
    def __init__(self, 
                 batch_size, 
                 device, 
                 dataset=None, 
                 train_dataset=None, 
                 valid_dataset=None, 
                 test_size=None) -> None:
        
        self.batch_size = batch_size
        self.device = device
        
        if train_dataset is not None and valid_dataset is not None:
            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
            
        else:
            assert dataset is not None and test_size is not None
            self.train_dataset, self.valid_dataset = self._train_test_split_dataset(dataset, test_size)
            

    def _train_test_split_dataset(self, dataset, test_size):
        train_dataset, valid_dataset = train_test_split(dataset, test_size = test_size, shuffle = False, random_state = 42)

        return train_dataset, valid_dataset
    
    def make_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle = True, worker_init_fn=np.random.seed(0))
        valid_dataloader = DataLoader(self.valid_dataset, self.batch_size, shuffle = False, worker_init_fn=np.random.seed(0))
        
        return train_dataloader, valid_dataloader
    
    def main_compute(self, loss_fn, emb_dim, object_num, n_epoch, lr, early_stopping=False, distance_metric = "euclidean", lamb = None, show_log=False, mapper=None):
        train_dataloader, valid_dataloader = self.make_dataloader()
        
        model = EmbeddingModel(emb_dim = emb_dim, object_num = object_num, mapper=mapper).to(self.device)
        
        model_training = ModelTraining(self.device, 
                                       model = model, 
                                       train_dataloader = train_dataloader, 
                                       valid_dataloader = valid_dataloader, 
                                       similarity = "pairwise", 
                                       distance_metric = distance_metric)
        
        test_loss, test_correct = model_training.main_compute(loss_fn=loss_fn, 
                                                lr=lr, 
                                                num_epochs=n_epoch, 
                                                early_stopping=early_stopping,
                                                lamb=lamb, 
                                                show_log=show_log)
        
        weights = model.state_dict()["Embedding.weight"].to('cpu').detach().numpy().copy()
        
        return weights, test_loss
    
    
class KFoldCV():
    def __init__(self, 
                 dataset, 
                 n_splits, 
                 search_space, 
                 study_name, 
                 results_dir,
                 batch_size,
                     device,
                     loss_fn,
                     emb_dim, 
                     object_num, 
                     n_epoch, 
                     lr, 
                     early_stopping,
                     distance_metric = "euclidean"
                     ) -> None:
        
        self.dataset = dataset
        self.full_data = [data for data in self.dataset]
        self.n_splits = n_splits
        self.search_space = search_space
        
        self.study_name = study_name
        self.save_dir = results_dir + "/" + study_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        #self.storage = "sqlite:///" + self.save_dir + "/" + study_name + ".db"
        self.storage = RDBStorage(
            "sqlite:///" + self.save_dir + "/" + study_name + ".db",
            engine_kwargs={"pool_size": 5, "max_overflow": 10},
            #retry_interval_seconds=1,
            #retry_limit=3,
            #retry_deadline=60
            )
        
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = loss_fn
        self.emb_dim = emb_dim
        self.object_num = object_num 
        self.n_epoch = n_epoch
        self.lr = lr
        self.early_stopping = early_stopping
        self.distance_metric = distance_metric
        
    #def training_for_one_split(self, train_indices, val_indices, lamb):
    #    
    #    train_dataset = [self.full_data[i] for i in train_indices]
    #    valid_dataset = [self.full_data[i] for i in val_indices]
    #    valid_dataloader = DataLoader(valid_dataset, self.batch_size, shuffle = False)
    #    train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle = True)
    #    
    #    model = EmbeddingModel(emb_dim = self.emb_dim, object_num = self.object_num).to(self.device)
    #
    #    model_training = ModelTraining(self.device, 
    #                                model = model, 
    #                                train_dataloader = train_dataloader, 
    #                                valid_dataloader = valid_dataloader, 
    #                                similarity = "pairwise", 
    #                                distance_metric = self.distance_metric)
    #
    #    loss, test_correct = model_training.main_compute(loss_fn=self.loss_fn, 
    #                                        lr=self.lr, 
    #                                        num_epochs=self.n_epoch, 
    #                                        early_stopping=self.early_stopping,
    #                                        lamb=lamb, 
    #                                        show_log=False)
    #    
    #    return loss
    #
    #def training(self, trial):
    #    kf = KFold(n_splits=self.n_splits, shuffle=True)
    #    
    #    lamb = trial.suggest_float("lamb", self.search_space[0], self.search_space[1], log=True)
    #    
    #    cv_losses = Parallel(n_jobs=-1)(delayed(self.training_for_one_split)(train_idx, val_idx, lamb) for train_idx, val_idx in kf.split(self.full_data))
    #    
    #    avg_cv_loss = sum(cv_losses) / len(cv_losses)
    #    
    #    return avg_cv_loss
        
    def training(self, trial):
        lamb = trial.suggest_float("lamb", self.search_space[0], self.search_space[1], log=True)
        
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        
        cv_loss = 0
        for train_indices, val_indices in kf.split(self.dataset):
            #train_dataset = Subset(self.dataset, train_indices)
            #valid_dataset = Subset(self.dataset, val_indices)
            train_dataset = [self.full_data[i] for i in train_indices]
            valid_dataset = [self.full_data[i] for i in val_indices]
            valid_dataloader = DataLoader(valid_dataset, self.batch_size, shuffle = False)
            train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle = True)
            
            
            model = EmbeddingModel(emb_dim = self.emb_dim, object_num = self.object_num).to(self.device)
        
            model_training = ModelTraining(self.device, 
                                       model = model, 
                                       train_dataloader = train_dataloader, 
                                       valid_dataloader = valid_dataloader, 
                                       similarity = "pairwise", 
                                       distance_metric = self.distance_metric)
        
            loss, test_correct = model_training.main_compute(loss_fn=self.loss_fn, 
                                                lr=self.lr, 
                                                num_epochs=self.n_epoch, 
                                                early_stopping=self.early_stopping,
                                                lamb=lamb, 
                                                show_log=False)
            
            cv_loss += loss
        cv_loss /= self.n_splits
        
        return cv_loss
    
    def optimize(self, n_trials):
        study = optuna.create_study(study_name = self.study_name, storage = self.storage, load_if_exists = True)
        study.optimize(self.training, n_trials=n_trials)

    def get_best_lamb(self, show_log=False):
        study = optuna.load_study(study_name = self.study_name, storage = self.storage)
        best_trial = study.best_trial
        df_trial = study.trials_dataframe()
        
        if show_log:
            sns.scatterplot(data = df_trial, x = "params_lamb", y = "value", s = 50)
            plt.xlabel("lamb")
            plt.xscale("log")
            plt.ylabel("cv loss")
            fig_path = os.path.join(self.save_dir, f"Optim_log.png")
            plt.savefig(fig_path)
            plt.tight_layout()
            plt.show()
            
        return best_trial.params["lamb"]
        
#%%
if __name__ == "__main__":
    ### Set parameters
    N_groups = 1
    
    data = "atyp" # "neutyp" or "atyp"
    
    if data == "neutyp":
        N_participant = 426
        data_dir = "../data/color_neurotypical/numpy_data/"
    elif data == "atyp":
        N_participant = 257
        data_dir = "../data/color_atypical/numpy_data/"

    participants_list = generate_random_grouping(N_participant=N_participant, 
                                                 N_groups=N_groups, 
                                                 seed=0)
    
    # device 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Train the model
    emb_dim = 20
    color_num = 93
    num_color_pairs = None # 75
    
    # Define the optimizer (e.g., Adam)
    lr = 0.01
    num_epochs = 100
    batch_size = 100
    early_stopping = False

    loss_fn = nn.MSELoss(reduction="sum")
    distance_metric = "euclidean"
    
    ### cv params
    n_splits = 5
    lamb_range = [1e-5, 1]
    study_name = f"{data} Ngroup={N_groups}"
    n_trials = 10
    

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

        ### cross validation
        #cv = KFoldCV(dataset=dataset(N_trials=num_color_pairs),
        #             n_splits=n_splits,
        #             search_space=lamb_range,
        #             study_name=study_name,
        #             results_dir="../results",
        #             batch_size=batch_size,
        #             device=device,
        #             loss_fn=loss_fn,
        #             emb_dim=emb_dim,
        #             object_num=color_num,
        #             n_epoch=num_epochs,
        #             lr=lr,
        #             early_stopping=early_stopping,
        #             distance_metric=distance_metric)
        #
        ##cv.optimize(n_trials=n_trials)
        #lamb = cv.get_best_lamb(show_log=True)
        lamb = 0.01
        
        ### main
        main_training = MainTraining(dataset = dataset(N_trials=num_color_pairs), 
                                     test_size = 1/n_splits, 
                                     batch_size = batch_size, 
                                     device = device)

        color_embeddings, loss = main_training.main_compute(loss_fn = loss_fn, 
                                   emb_dim = emb_dim, 
                                   object_num = color_num, 
                                   n_epoch = num_epochs, 
                                   early_stopping=early_stopping,
                                   lr = lr, 
                                   distance_metric = distance_metric,
                                   lamb=lamb)

        rearranged_color_embeddings_list.append(color_embeddings[reorder_idxs, :])
    np.save(f"../results/rearranged_embeddings_list_{data}_Ngroup={N_groups}.npy", rearranged_color_embeddings_list)

    # %%    
    #### check dimensionality
    #embeddings = rearranged_color_embeddings_list[0]
    #plt.hist(np.abs(embeddings).flatten())
    #plt.show()
    #
    #plt.hist(np.max(np.abs(embeddings), axis=0))
    #plt.show()
    # %%
    ## Load the data file
    #num_response = []
    #num_non_Nan = []
    #for i in range(257):
    #    data_dir = "../data/color_atypical/numpy_data/"
    #    filepath = f"participant_{i}.npy"
    #    data = np.load(data_dir + filepath, allow_pickle=True)
    #    data = data.astype(np.float64)
#
    #    # count the number of non Nan elements
    #    non_nan = np.count_nonzero(~np.isnan(data))
    #    num_non_Nan.append(non_nan)
#
    #    # Get lower triangle indices where data is not NaN
    #    lower_triangle_indices = np.tril_indices(data.shape[0], -1)  # -1 excludes the diagonal
    #    values = data[lower_triangle_indices]
    #    non_nan_indices = np.where(~np.isnan(values))
#
    #    # Get final indices and values
    #    final_indices = (lower_triangle_indices[0][non_nan_indices], lower_triangle_indices[1][non_nan_indices])
    #    final_values = values[non_nan_indices]
#
    #    X = list(zip(*final_indices)) # Zip the indices to get (row, col) pairs
    #    y = list(final_values)
#
    #    num_response.append(len(y))
    ## %%
    #print(num_non_Nan)
    #print(num_response)
# %%
