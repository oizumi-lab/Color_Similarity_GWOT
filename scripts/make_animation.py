"""
Make gif animation for aligned emneddings

Genji Kawakita
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
import pickle as pkl
from src.utils import *
from src.analysis_utils import *
from src.preprocess_utils import *

def gif_animation(embedding_list,markers_list,colors,fig_size=(15,12),save_anim=False,save_path=None):


    X = embedding_list[0] # referenced embeddings

    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.add_subplot(111, projection="3d")
    #ax.set_axis_off() 
    n_groups = len(embedding_list)
    for grp_idx in range(n_groups):
        ax.scatter(xs=embedding_list[grp_idx][:,0],ys=embedding_list[grp_idx][:,1],zs=embedding_list[grp_idx][:,2],\
                marker=markers_list[grp_idx],color=colors,alpha=1,s=100,label=f"Group {grp_idx+1}")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlim()
    ax.set_ylim()
    ax.set_zlim()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    ax.axes.get_zaxis().set_visible(True)
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    ax.set_facecolor('white')
    ax.grid(True)
    #bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
    #plt.tight_layout()

    # Create the animation
    def update(frame):
        ax.view_init(elev=10,azim=frame*1)

    anim = FuncAnimation(fig, update, frames=range(180), repeat=False,interval=150)

    # Save the animation as a gif
    if save_anim:
        anim.save(save_path, dpi=80, writer="pillow")

    # Show the animation
    plt.show()

# %% load data
# load color-atypical data
data_dir = "../results/optimization/"
mat_dir = "../results/diss_mat"
atypical_path = os.path.join(mat_dir,"reordered_diss_mat_color_atypical_seed_0_fill_val_3.5.npy")
# load color-atypical dissimilarity matrix
atypical_mat = np.load(atypical_path,allow_pickle=True)

# load color-neurotypical dissimilarity matrix
neurotypical_path = os.path.join(mat_dir,"reordered_diss_mat_num_groups_5_seed_0_fill_val_3.5.pickle")

with open(neurotypical_path, "rb") as input_file:
    data = pkl.load(input_file)
neurotypical_matrices = data["group_ave_mat"]

# concatenate matrices
temp= np.expand_dims(atypical_mat, axis=0)

# Concatenate the matrices along the first dimension (axis=0)
all_matrices = np.concatenate((neurotypical_matrices, temp), axis=0)
n_groups = all_matrices.shape[0]

# load colors
new_color_order = list(np.load('../data/hex_code/new_color_order.npy')) # color codes
new_color_dict = make_color_idx_dict(new_color_order)

# %%
# apply MDS to all groups
embedding_list = []
for i in range(n_groups):
    embedding = MDS_embedding(all_matrices[i,:,:],n_dim=3,random_state=5)
    embedding_list.append(embedding)

best_OT_mat = np.empty((n_groups,n_groups),dtype="O")
for i in range(n_groups):
    for j in range(i+1,n_groups):
        data = np.load(os.path.join(data_dir,f"optimization_results_Group{i}_Group{j}_n=5_ntrials=10_epsilon=0.02to0.2_n_eps=20.npy"),allow_pickle=True)
        _,gwd_min_trial = get_min_gwd(data)
        OT = data[gwd_min_trial]["OT"]
        best_OT_mat[i,j] = OT

# apply procrstes to align Group 2-5 embeddings (Y_i) to Group 1 (X)
nn_embedding_list = []
nn_embedding_list.append(embedding_list[0])
for i in range(1,n_groups-1):
    X = embedding_list[0]
    Y = embedding_list[i]
    OT = best_OT_mat[0,i]
    _, aligned_embedding = procrustes_alignment(X,Y,OT)
    nn_embedding_list.append(aligned_embedding)

# apply procrstes to align Group 6 embeddings (Y_6) to Group 1 (X)
na_embedding_list = []
X = embedding_list[0]
na_embedding_list.append(X)
Y = embedding_list[5]
OT = best_OT_mat[0,5]
_, aligned_embedding = procrustes_alignment(X,Y,OT)
na_embedding_list.append(aligned_embedding)
# %% neurotypical-atypical alignment
save_dir = "../results/figs"
save_path = os.path.join(save_dir,"Supp_anim_neurotypical_atypical.gif")
markers_list = ["o","X"]
gif_animation(na_embedding_list,markers_list,new_color_order,fig_size=(12,9),save_anim=True,save_path=save_path)

# %% neurotypical-neurotypical alignment
save_path = os.path.join(save_dir,"Supp_anim_neurotypical_neurotypical.gif")
markers_list = ["o","X","^","s",">"]
gif_animation(nn_embedding_list,markers_list,new_color_order,fig_size=(12,9),save_anim=True,save_path=save_path)

# %%
