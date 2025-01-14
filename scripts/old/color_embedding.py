#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import itertools

from src.preprocess_utils import *
from src.utils import *
from src.plot_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#%%
# Create empty lists to store X and y
X = []
y = []
data_list = []
N_participant = 426

# Loop over the file names
for i in range(N_participant):
    # Load the data file
    nt_npy_dir = "../data/color_neurotypical/numpy_data/"
    filepath = f"participant_{i}.npy"
    data = np.load(nt_npy_dir+filepath, allow_pickle=True)
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
    X.extend(list(zip(*final_indices)))  # Zip the indices to get (row, col) pairs
    y.extend(final_values)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.LongTensor(X)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)
batch_size = 124

# Create a dataloader
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

# %% Draw whole-group-average dissimilarity matrix for color-neurotypical participants

# load all color-neurotypical data
nt_npy_dir = "../data/color_neurotypical/numpy_data"
nt_npy_list = os.listdir(nt_npy_dir) 
nt_all_data = load_all_participant_npy(nt_npy_dir,nt_npy_list)

# make the whole-group-average dissimilarity matrix
ave_mat = average_matrix(nt_all_data)

# rearrange the matrix
# load color codes
old_color_order = list(np.load('../data/hex_code/original_color_order.npy'))
new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
# get the reordering indices
reorder_idxs = get_reorder_idxs(old_color_order,new_color_order)
ave_mat = rearrange_diss_mat(ave_mat,reorder_idxs)

# 
cmap = cmap_discretize(plt.cm.gray, 8)
save_dir = "../results/figs"
plt.figure(figsize=(20,20))
sns.heatmap(ave_mat,cbar=False,cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'{save_dir}/fig1c_diss_mat.png',dpi=300)
plt.show()

#%%
# Load your similarity scores
option_average = False
if option_average:
    similarity_data = torch.tensor(ave_mat, dtype=torch.float32)
 
    color_num = 93
    pairs = list(itertools.combinations(range(color_num), 2))
    rating_list = []
    for pair in pairs:
        rating = similarity_data[pair[0], pair[1]]
        rating_list.append(rating)

    X = torch.LongTensor(pairs)
    y = torch.tensor(rating_list, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    batch_size = 100
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

#%%
def compute_similarity(x):
    ## size of x (batch x 2 x emb_dim)
    # dot product
    # sim = torch.sum(torch.mul(x[:, 0, :], x[:, 1, :]), dim = 1) #[batch,1]
        
    # euclidean distance
    sim = torch.norm(x[:, 0, :] - x[:, 1, :], p = 2, dim = 1)
    return sim

#%%
# device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a simple linear function to map color embeddings to similarity scores
class Model(nn.Module):
    def __init__(self, emb_dim = 3, color_num = 93):
        super(Model, self).__init__()
        self.Embedding = nn.Embedding(color_num, emb_dim)

    def forward(self, x):
        x = self.Embedding(x)
        return x    

emb_dim = 3
color_num = 93
model = Model(emb_dim=emb_dim, color_num=color_num)
model.to(device)

# Define the loss function (e.g., mean squared error for regression)
loss_fn = nn.MSELoss()

# Define the optimizer (e.g., Adam)
lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=lr)
#%%
# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X_emb = model(X)
        pred = compute_similarity(X_emb)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# The optimized color embeddings are now in the 'embeddings' tensor

# %%
# color_embeddings = model.state_dict()["Embedding.weight"].to('cpu').detach().numpy().copy()

color_embeddings = []
for color_ind in torch.arange(color_num):
    color_embeddings.append(model(color_ind.to(device)).to('cpu').detach().numpy().copy())
color_embeddings = np.array(color_embeddings)

#%%
# Create a PCA instance
pca = PCA(n_components=3)
# Fit the data to the PCA model
pca.fit(color_embeddings)
# Transform the data to the principal components
X_pca = pca.transform(color_embeddings)

fig= plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2],
#        marker="o", color=new_color_order, alpha=1)
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2],
        marker="o", color=old_color_order, alpha=1)
plt.show()
# %% check the prediction of the model
data_sim_list = []
model_sim_list = []
for pair in pairs:
    data_sim = ave_mat[pair[0], pair[1]]
    data_sim_list.append(data_sim)
    
    x1 = color_embeddings[pair[0],:]
    x2 = color_embeddings[pair[1],:]
    model_sim = np.linalg.norm(x1-x2)
    model_sim_list.append(model_sim)
    
df = pd.DataFrame(data={'data': data_sim_list, 'model': model_sim_list})
# %%
