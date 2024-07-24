from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
import numpy as np
import random
import pandas as pd

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = False
    
    
class ComputeSimilarityTrip:
    def __init__(self, distance_metric) -> None:
        self.distance_metric = distance_metric
        pass
    
    def cosine_similarity(self, x, T = 20):
        dot_12 = T * F.cosine_similarity(x[:, 0, :], x[:, 1, :], dim = 1).unsqueeze(1)
        dot_23 = T * F.cosine_similarity(x[:, 1, :], x[:, 2, :], dim = 1).unsqueeze(1)
        dot_13 = T * F.cosine_similarity(x[:, 0, :], x[:, 2, :], dim = 1).unsqueeze(1)
        output = torch.cat([dot_23, dot_13, dot_12], dim = 1) #[batch, 3]
        return output
        
    def dot(self, x):
        dot_12 = torch.sum(torch.mul(x[:, 0, :], x[:, 1, :]), dim = 1).unsqueeze(1) #[batch,1]
        dot_23 = torch.sum(torch.mul(x[:, 1, :], x[:, 2, :]), dim = 1).unsqueeze(1)
        dot_13 = torch.sum(torch.mul(x[:, 0, :], x[:, 2, :]), dim = 1).unsqueeze(1)
        output = torch.cat([dot_23, dot_13, dot_12], dim = 1) #[batch, 3]
        return output

    def euclidean(self, x):
        dist_12 = torch.sqrt(torch.norm(x[:, 0, :] - x[:, 1, :], p = 2, dim = 1)).unsqueeze(1)
        dist_23 = torch.sqrt(torch.norm(x[:, 1, :] - x[:, 2, :], p = 2, dim = 1)).unsqueeze(1)
        dist_13 = torch.sqrt(torch.norm(x[:, 0, :] - x[:, 2, :], p = 2, dim = 1)).unsqueeze(1)
        output = torch.cat([-dist_23, -dist_13, -dist_12], dim = 1) #[batch, 3]
        return output

    def __call__(self, x : torch.Tensor, T = 20):
        if self.distance_metric == "cosine":
            return self.cosine_similarity(x, T)
        elif self.distance_metric == "dot":
            return self.dot(x)
        elif self.distance_metric == "euclidean":
            return self.euclidean(x)


class ComputeSimilarityPairwise:
    def __init__(self, distance_metric) -> None:
        self.distance_metric = distance_metric
        pass
    
    def euclidean(self, x):
        ## size of x (batch x 2 x emb_dim)
        # euclidean distance
        sim = torch.norm(x[:, 0, :] - x[:, 1, :], p = 2, dim = 1)
        return sim    
    
    def dot(self, x):
        # dot product
        sim = torch.sum(torch.mul(x[:, 0, :], x[:, 1, :]), dim = 1) #[batch,1]
        return sim
    
    def __call__(self, x : torch.Tensor):
        if self.distance_metric == "dot":
            return self.dot(x)
        elif self.distance_metric == "euclidean":
            return self.euclidean(x)
    
    
class EmbeddingModel(nn.Module):
    def __init__(self, emb_dim, object_num, init_fix = True):
        if init_fix:
            torch_fix_seed()
        super(EmbeddingModel, self).__init__()
        self.Embedding = nn.Embedding(object_num, emb_dim) 
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            #module.weight.data.uniform_(0, 1)
            module.weight.data.normal_(mean=1.0, std=0.1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x : torch.Tensor):
        x = self.Embedding(x)
        return x
         
    
class ModelTraining():
    def __init__(self, device, model, train_dataloader, valid_dataloader, similarity = "pairwise", distance_metric = "euclidean") -> None:
        self.device = device
        self.model = model
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        if similarity == "pairwise":
            self.compute_similarity = ComputeSimilarityPairwise(distance_metric)
        elif similarity == "triplet":
            self.compute_similarity = ComputeSimilarityTrip(distance_metric)
        
    def run_epoch_computation(self, dataloader, loss_fn, optimizer, lamb = None):
        size = len(dataloader.dataset)
        running_loss, correct = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            sim = self.compute_similarity(pred)
            loss = loss_fn(sim, y)
            
            if self.model.training == True:
                if lamb is not None:
                    #l1 normalization
                    l1 = 0
                    for w in self.model.parameters():
                        l1 += torch.sum(torch.abs(w))
                    loss += lamb * l1
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()   
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()  

        running_loss /=  size
        correct /= size
        
        return running_loss, correct
    
    def main_compute(self, loss_fn, lr, num_epochs, early_stopping=True, lamb = None, show_log = True):
        #loss_fn = nn.CrossEntropyLoss(reduction = "sum")
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        
        training_loss = list()
        testing_loss = list()
        
        best_test_loss = float("inf")
        early_stop_counter = 0
        patience = 5

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            train_loss, train_correct = self.run_epoch_computation(self.train_dataloader, loss_fn, optimizer, lamb)
            
            with torch.no_grad():
                self.model.eval()
                test_loss, test_correct = self.run_epoch_computation(self.valid_dataloader, loss_fn, optimizer, lamb)
            
            if early_stopping:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    early_stop_counter = 0
                
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print("Early stopping triggered. Training stopped.")
                        break
                    
            if show_log: 
                print('[%d/%5d] \ntrain loss: %.5f, accuracy: %.2f \ntest loss: %.5f, accuracy: %.2f' % (epoch + 1, num_epochs, train_loss, train_correct * 100, test_loss, test_correct * 100))
                training_loss.append(train_loss)
                testing_loss.append(test_loss)
        
        if show_log:
            plt.close()
            plt.figure()
            plt.subplot(211)
            plt.plot(training_loss)
            plt.subplot(212)
            plt.plot(testing_loss)
            plt.show()
        
        return test_loss, test_correct
    
    
class MakeDataset_THINGS():
    def __init__(self, trip, category, n_group, triplets_size = None, shuffle = False) -> None:
        self.trip = trip
        self.category = category
        self.n_group = n_group
        self.shuffle = shuffle
        
        if self.category == "gender":
            self.category_names = ["male", "female"]
        elif self.category == "age":
            self.category_names = ["under", "over"]
        elif self.category == "both":
            self.category_names = ["male&under", "male&over", "female&under", "female&over"]
        elif self.category == "shuffle":
            self.category_names = ["group"]
        
        self.preprocessing_triplets(triplets_size) 
        
        pass
    
    def preprocessing_triplets(self, triplets_size = None):
        self.trip.loc[:, "image1" : "choice"] -= 1 
        if self.category == "age":
            self.trip = self.trip.dropna(how = "any", axis = 0) # Remove subjects whose "age" are NaN
            self.trip = self.trip.sort_values(["age", "subject_id"])
            
            print("Under : under", self.trip["age"].iloc[triplets_size], ", Over : over", self.trip["age"].iloc[- triplets_size])
            
            self.trip.loc[: self.trip.index[triplets_size], "age"] = "under"
            self.trip.loc[self.trip.index[len(self.trip) - triplets_size] : , "age"] = "over"
        
    def get_name_list(self):
        name_list = list()
        for category_name in self.category_names:
            for i in range(self.n_group):
                name_list.append(f"{category_name}_{i + 1}")
                
        return name_list

    def split_data_for_categories(self):
        trip_list = list()
        # extract a particular category
        for category_name in self.category_names:
            if self.category == "shuffle":
                trip_category = self.trip.sort_values(["subject_id"])
            else:
                # Sort trilets with subject_ids or not
                if self.shuffle is False:
                    self.trip = self.trip.sort_values(["subject_id"])
                else:
                    self.trip = self.trip.sample(frac = 1)
                trip_category = self.trip[pd.Series(self.trip[self.category] == category_name)] 

            # split
            trip_groups = np.array_split(trip_category, self.n_group)

            # Remove duplicated subjects
            if self.shuffle is False:
                for i in range(self.n_group - 1):
                    sub = set(trip_groups[i]["subject_id"]) & set(trip_groups[i + 1]["subject_id"])
                    if sub != set():
                        trip_groups[i] = trip_groups[i][pd.Series(trip_groups[i]["subject_id"] != list(sub)[0])]
            trip_list += trip_groups

        # Arrange the length
        min_length = np.min([len(trip_list[i]) for i in range(len(trip_list))])
        trip_list = [trip_list[i][:min_length] for i in range(len(trip_list))]
        print("Number of triplets : ", min_length)
        
        # Check subject numbers
        #for trip_group in trip_list:
        #    print(len(set(trip_group["subject_id"])))

        return trip_list

    def make_dataset(self):
        trip_list = self.split_data_for_categories()
        dataset_list = list()
        for dataset in trip_list:
            data = torch.LongTensor(dataset.loc[:, "image1" : "image3"].values)
            label = torch.LongTensor(dataset.loc[:, "choice"].values)
            dataset = TensorDataset(data, label)
            dataset_list.append(dataset)

        return dataset_list