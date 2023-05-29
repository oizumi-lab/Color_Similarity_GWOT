"""
Make Fig. 3

Genji Kawakita
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *
from src.preprocess_utils import *
from src.plot_utils import *
from src.analysis_utils import *

# %% Fig 3.a (Dissimilarity matrices for 5 groups)
# load all data
data_path = "../results/diss_mat/reordered_diss_mat_num_groups_5_seed_0_fill_val_3.5.pickle"

with open(data_path, "rb") as input_file:
    data = pkl.load(input_file)
matrices = data["group_ave_mat"]
n_groups = matrices.shape[0]


# plot and save figs
save_dir = "../results/figs"
cmap = cmap_discretize("gist_gray",8)
#cmap = "gist_gray"
for i in range(n_groups):
    plt.figure(figsize=(20,20))
    sns.heatmap(matrices[i,:,:],cbar=False,cmap=cmap,vmin=0, vmax=7)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig3a_group{i+1}.png',dpi=300)
    plt.show()
# %% Fig 3.b (Optimal transportation plans)
data_dir = "../results/optimization"
fig_ot_dir = "../results/figs/ot"

# get the reordering indices
max_val = 1/93

# list of best OTs
best_OTs = []

for i in range(1,5):
    data = np.load(os.path.join(data_dir,f"optimization_results_Group0_Group{i}_n=5_ntrials=10_epsilon=0.02to0.2_n_eps=20.npy"),allow_pickle=True)
    _,gwd_min_trial = get_min_gwd(data)
    fig,ax = plt.subplots(1,1,figsize=(20,20))
    OT = data[gwd_min_trial]["OT"]
    a = sns.heatmap(OT,ax=ax,cbar=False,vmax=max_val) 
    plt.xticks([])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    #plt.savefig(f'{fig_ot_dir}/OT_Group{i+1}_Group{j+1}.png',dpi=300)
    best_OTs.append(OT)

# %% Fig 3.c MDS plot for each group

# apply MDS to each dissimilarity matrix
markers_list = ["o","X","^","s",">"]
embedding_list = []
new_color_order = list(np.load('../data/hex_code/new_color_order.npy')) # color codes
new_color_dict = make_color_idx_dict(new_color_order)
# plot MDS embedding
for i in range(n_groups):
    embedding = MDS_embedding(matrices[i,:,:],n_dim=3,random_state=5)
    save_name = f"fig3c_group{i+1}_mds.png"
    plot_MDS_embedding(embedding,markers_list[i],\
                       new_color_order,alpha=1,s=300,fig_size=(15,15),save=False,\
                        fig_dir=save_dir,save_name=save_name)
    embedding_list.append(embedding)


# %% Fig 3.d (Aligned embeddings)
# apply procrstes to align Group 2-5 embeddings (Y_i) to Group 1 (X)
aligned_embedding_list = []
for i in range(1,n_groups):
    X = embedding_list[0]
    Y = embedding_list[i]
    OT = best_OTs[i-1]
    _, aligned_embedding = procrustes_alignment(X,Y,OT)
    aligned_embedding_list.append(aligned_embedding)

# plot aligned embeddings
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection="3d")
for grp_idx in range(n_groups):
    if grp_idx == 0:
        ax.scatter(xs=embedding_list[grp_idx][:,0],ys=embedding_list[grp_idx][:,1],zs=embedding_list[grp_idx][:,2],\
            marker=markers_list[grp_idx],color=new_color_order,alpha=1,s=300,label=f"Group {grp_idx+1}")
    else:
        ax.scatter(xs=aligned_embedding_list[grp_idx-1][:,0],ys=aligned_embedding_list[grp_idx-1][:,1],zs=aligned_embedding_list[grp_idx-1][:,2],\
            marker=markers_list[grp_idx],color=new_color_order,alpha=1,s=300,label=f"Group {grp_idx+1}")
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_zaxis().set_visible(False)
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.set_facecolor('white')
ax.view_init(elev=15, azim=150)
ax.legend(fontsize=24,loc=(0.8,0.6))
leg = ax.get_legend()
for i in range(n_groups):
    leg.legendHandles[i].set_color('k')
plt.tight_layout()
plt.savefig(os.path.join(save_dir,"fig3d_aligned_embeddings.png"))
plt.show()
    
# %% Calculate k-nearest color matching rate between all the pairs of color-neurotypical groups

# get the optimal transport plan between all the pairs
best_OT_mat = np.empty((n_groups,n_groups),dtype="O")
for i in range(n_groups):
    for j in range(i+1,n_groups):
        data = np.load(os.path.join(data_dir,f"optimization_results_Group{i}_Group{j}_n=5_ntrials=10_epsilon=0.02to0.2_n_eps=20.npy"),allow_pickle=True)
        _,gwd_min_trial = get_min_gwd(data)
        OT = data[gwd_min_trial]["OT"]
        best_OT_mat[i,j] = OT

k_list = [1,3,5]
k_matching_rates = []
for k in k_list:
    temp_rates = []
    for i in range(n_groups):
        for j in range(i+1,n_groups):
            X = embedding_list[i]
            Y = embedding_list[j]
            OT = best_OT_mat[i,j]
            _,aligned_embedding = procrustes_alignment(X,Y,OT)
            # calculate k-nearest color matching rate
            # compute distance matrix
            dist_mat = compute_distance_matrix(X,aligned_embedding)
            _,k_matching_rate = k_nearest_colors_matching_rate(dist_mat, new_color_dict, new_color_dict, k)
            temp_rates.append(k_matching_rate)
    k_matching_rates.append(temp_rates)
# average matching rates
print(f"N-N k=1: {np.mean(k_matching_rates[0])}")
print(f"N-N k=3: {np.mean(k_matching_rates[1])}")
print(f"N-N k=5: {np.mean(k_matching_rates[2])}")


# %%
