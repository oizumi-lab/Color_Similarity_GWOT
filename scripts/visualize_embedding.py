#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#%%
def plot_3d_with_pca(data, colors, elev=30, azim=45, plot=False):
    """
    Perform PCA on data and plot the first 3 components in 3D.
    
    Parameters:
    - data (np.ndarray): The input data with shape (n, d)
    - colors (list): A list of colors (one for each data point)
    """
    
    # Assert that colors list length matches data length
    assert len(colors) == len(data), "Number of colors must match number of data points."
    
    # Perform PCA and reduce to 3 components
    pca = PCA(n_components=data.shape[1])
    reduced_data = pca.fit_transform(data)
    
    # Compute cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Number of original dimensions
    original_dims = data.shape[1]
    
    # Plot
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, original_dims + 1), cumulative_explained_variance, marker='o', label='Cumulative Explained Variance')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')  # Optional: 95% threshold line
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Cumulative Explained Variance vs Original Dimensions')
        plt.xlim([0, original_dims])
        plt.grid(True)
        plt.legend()
        plt.show()
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=colors)
    
    # Label the axes
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Set the viewing angle
    ax.view_init(elev, azim) 
    
    plt.tight_layout()
    plt.show()
    
def plot_3d_tsne(data, colors, elev=30, azim=45):
    """
    Apply t-SNE on data and plot the result in 3D.
    
    Parameters:
    - data (np.ndarray): The input data with shape (n, d)
    - colors (list): A list of colors (one for each data point)
    """
    
    # Assert that colors list length matches data length
    assert len(colors) == len(data), "Number of colors must match number of data points."
    
    # Apply t-SNE and reduce to 3 components
    tsne = TSNE(n_components=3, random_state=42)
    embedded_data = tsne.fit_transform(data)
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], c=colors)
    
    # Set the viewing angle
    ax.view_init(elev, azim) 
    
    plt.show()

#%%
data = "neutyp"
emb_dim = 20
Z = 426
N_groups = 1
N_trials = 75
N_sample = 1

embedding_pairs = np.load(f"../results/embeddings_pairs_list_{data}_emb={emb_dim}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}.npy")

#%%
embedding = embedding_pairs[0][0]
colors = list(np.load('../data/hex_code/original_color_order.npy'))
# %%
for i in range(18):
    plot_3d_with_pca(data=embedding, colors=colors, elev=45, azim=i*10)
#plot_3d_tsne(data=embedding, colors=colors, elev=30, azim=45)
# %%
