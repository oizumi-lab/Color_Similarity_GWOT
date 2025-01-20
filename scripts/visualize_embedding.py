#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from matplotlib.animation import FuncAnimation

#%%
def pca(data, colors, elev=30, azim=45, plot=False):
    """
    Perform PCA on data and plot the first 3 components in 3D.
    
    Parameters:
    - data (np.ndarray): The input data with shape (n, d)
    - colors (list): A list of colors (one for each data point)
    """
    
    # Assert that colors list length matches data length
    assert len(colors) == len(data), "Number of colors must match number of data points."
    
    # Perform PCA and reduce to 3 components
    if data.shape[1] > 3:
        pca = PCA(n_components=data.shape[1])
        reduced_data = pca.fit_transform(data)
        reduced_data = reduced_data[:, :3]
    else:
        reduced_data = data
    
    # Compute cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # scree plot
    # get the eigenvalues
    eigvals = pca.explained_variance_
    
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
        
        # scree plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, original_dims + 1), eigvals, marker='o', label='Cumulative Explained Variance')
        #plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')  # Optional: 95% threshold line
        plt.xlabel('Number of Components')
        plt.ylabel('Eigenvalues')
        plt.title('PCA Scree Plot')
        plt.xlim([0, original_dims])
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return reduced_data

def plot_3d(data, colors, elev=30, azim=45, marker='o', save_dir=None):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, marker=marker, alpha=1, s=50)
    
    # Label the axes
    ax.set_xlabel('PC1', labelpad=-10, fontsize=25)
    ax.set_ylabel('PC2', labelpad=-10, fontsize=25)
    ax.set_zlabel('PC3', labelpad=-10, fontsize=25)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis._axinfo["grid"].update({"color": "black", "linewidth": 0.5, "alpha": 0.3})
    ax.yaxis._axinfo["grid"].update({"color": "black", "linewidth": 0.5, "alpha": 0.3})
    ax.zaxis._axinfo["grid"].update({"color": "black", "linewidth": 0.5, "alpha": 0.3})
    ax.grid(True, color='black', linewidth=0.5, alpha=0.3)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    ax.axes.get_zaxis().set_visible(True)
    
    
    # Set the viewing angle
    ax.view_init(elev, azim) 
    
    #ax.patch.set_alpha(0)
    
    plt.tight_layout()
    plt.show()
    
    if save_dir is not None:
        fig.savefig(save_dir, transparent=True)

def plot_3d_ovelay(data1, data2, colors1, colors2, name1, name2, elev=30, azim=45, marker1='o', marker2='x', save_dir=None, animation_save_path=None):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c=colors1, label=name1, marker=marker1, alpha=1, s=50)
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c=colors2, label=name2, marker=marker2, alpha=1, s=50)
    
    # Label the axes
    ax.set_xlabel('PC1', labelpad=-10, fontsize=25)
    ax.set_ylabel('PC2', labelpad=-10, fontsize=25)
    ax.set_zlabel('PC3', labelpad=-10, fontsize=25)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis._axinfo["grid"].update({"color": "black", "linewidth": 0.5, "alpha": 0.3})
    ax.yaxis._axinfo["grid"].update({"color": "black", "linewidth": 0.5, "alpha": 0.3})
    ax.zaxis._axinfo["grid"].update({"color": "black", "linewidth": 0.5, "alpha": 0.3})
    ax.grid(True, color='black', linewidth=0.5, alpha=0.3)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    ax.axes.get_zaxis().set_visible(True)
    
    
    # Set the viewing angle
    ax.view_init(elev, azim) 
    
    # Add a legend
    ax.legend()
    # set the font size of the legend
    plt.setp(ax.get_legend().get_texts(), fontsize='25')
    
    ax.patch.set_alpha(0)
    
    plt.tight_layout()
    plt.show()
    
    if save_dir is not None:
        fig.savefig(save_dir, transparent=True)
    
    if animation_save_path is not None:
        # make animation
        def update(frame):
            ax.view_init(elev, frame)
            return fig,

        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=True)

        ani.save(animation_save_path, writer='ffmpeg', fps=20)
    
    
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

def procrustes(
        embedding_target: np.ndarray,
        embedding_source: np.ndarray,
        OT: np.ndarray
    ) -> np.ndarray:
        """Function that brings embedding_source closest to embedding_target by orthogonal matrix

        Args:
            embedding_target (np.ndarray):
                Target embeddings with shape (n_target, m).

            embedding_source (np.ndarray):
                Source embeddings with shape (n_source, m).

            OT (np.ndarray):
                Transportation matrix from source to target with shape (n_source, n_target).


        Returns:
            new_embedding_source (np.ndarray):
                Transformed source embeddings with shape (n_source, m).
        """
        U, S, Vt = np.linalg.svd(np.matmul(embedding_source.T, np.matmul(OT, embedding_target)))
        Q = np.matmul(U, Vt)
        new_embedding_source = np.matmul(embedding_source, Q)
        return new_embedding_source

def MDS_from_RDM(embedding, colors, plot=False):
    # Compute the distance matrix
    distance_matrix = np.zeros((embedding.shape[0], embedding.shape[0]))
    for i in range(embedding.shape[0]):
        for j in range(embedding.shape[0]):
            distance_matrix[i, j] = np.linalg.norm(embedding[i] - embedding[j])
    
    # Perform MDS
    mds = MDS(n_components=3, dissimilarity='precomputed')
    reduced_data = mds.fit_transform(distance_matrix)
    
    # Plot
    if plot:
        plot_3d(reduced_data, colors)
    
    return reduced_data

#%%
for data in ["neutyp", "atyp", "n-a"]:
    emb_dim = 20
    Z = 128
    N_groups = 1
    N_trials = 75
    N_sample = 1
    i = 0

    # Load the embedding
    file_dir = f"../results/{data}/Z={Z}/seed{i}/"
    embedding_1 = np.load(file_dir + f"embedding_{data}_0.npy")
    embedding_2 = np.load(file_dir + f"embedding_{data}_1.npy")
    OT = np.load(file_dir + f"OT_{data}.npy")

    colors = list(np.load('../data/hex_code/original_color_order.npy'))
    
    if data == "neutyp":
        name1, name2 = "Group T1", "Group T2"
    elif data == "atyp":
        name1, name2 = "Group A1", "Group A2"
    else:
        name1, name2 = "Group T1", "Group A1"

    # plot
    reduced_data_1 = pca(embedding_1, colors, plot=True)
    reduced_data_2 = pca(embedding_2, colors, plot=True,)
    plot_3d(reduced_data_1, colors, save_dir=os.path.join(file_dir, f"embedding_{data}_0.png"))
    plot_3d(reduced_data_2, colors, marker='s', save_dir=os.path.join(file_dir, f"embedding_{data}_1.png"))


    # align
    aligned_embedding_2 = procrustes(reduced_data_1, reduced_data_2, OT)
    plot_3d_ovelay(
        reduced_data_1, 
        aligned_embedding_2[0], 
        colors, colors, 
        name1, name2, 
        marker2='s', 
        save_dir=os.path.join(file_dir, f"aligned_embedding_{data}.png"),
        animation_save_path=os.path.join(file_dir, f"aligned_embedding_{data}.mp4")
        )

# %%
