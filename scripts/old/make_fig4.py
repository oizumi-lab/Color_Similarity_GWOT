"""
Make Fig. 4

Genji Kawakita
"""
# %% Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from src.utils import *
from src.utils.preprocess_utils import *
from src.utils.plot_utils import *
from src.utils.analysis_utils import *

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
# %% Fig 4.a (Dissimilarity matrices of color-neurotypical (Group 1) and color-atypical groups)

# 
cmap = cmap_discretize(plt.cm.gray, 8)
save_dir = "../results/figs"
plt.figure(figsize=(20,20))
sns.heatmap(atypical_mat,cbar=False,cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'{save_dir}/fig4a_atypical_diss_mat.png',dpi=300)
plt.show()
# %% Fig 4.b (Alignment between color-neurotypical and color-atypical groups)
# run plot_optimal_transportation_plans.py (OT between Group 1 & 6 is used in the fig)

# %% Fig 4.c k-nearest matching rate

# apply MDS to all groups
embedding_list = []
for i in range(n_groups):
    embedding = MDS_embedding(all_matrices[i,:,:],n_dim=3,random_state=5)
    embedding_list.append(embedding)


# %%
best_OT_mat = np.empty((n_groups,n_groups),dtype="O")
for i in range(n_groups):
    for j in range(i+1,n_groups):
        data = np.load(os.path.join(data_dir,f"optimization_results_Group{i}_Group{j}_n=5_ntrials=10_epsilon=0.02to0.2_n_eps=20.npy"),allow_pickle=True)
        _,gwd_min_trial = get_min_gwd(data)
        OT = data[gwd_min_trial]["OT"]
        best_OT_mat[i,j] = OT

k_list = [1,3,5]
df = pd.DataFrame(columns=['matching_rate', 'pairs', 'k'])
nn_rates = [] # N-N matching rates
na_rates = [] # N-A matching rates
for k in k_list:
    nn = []
    for i in range(n_groups-1):
        for j in range(i+1,n_groups-1):
            X = embedding_list[i]
            Y = embedding_list[j]
            OT = best_OT_mat[i,j]
            _,aligned_embedding = procrustes_alignment(X,Y,OT)
            dist_mat = compute_distance_matrix(X,aligned_embedding)
            _,k_matching_rate = k_nearest_colors_matching_rate(dist_mat, new_color_dict, new_color_dict, k)
            nn.append(k_matching_rate)
    for i in range(len(nn)):
        row = {'matching_rate': nn[i], 'pairs': "nn", 'k': k}
        # Append the new row to the DataFrame
        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
    na = []
    for i in range(n_groups-1):
        X = embedding_list[i]
        Y = embedding_list[5] # atypical
        OT = best_OT_mat[i,5]
        _,aligned_embedding = procrustes_alignment(X,Y,OT)
        dist_mat = compute_distance_matrix(X,aligned_embedding)
        _,k_matching_rate = k_nearest_colors_matching_rate(dist_mat, new_color_dict, new_color_dict, k)
        na.append(k_matching_rate)
    for i in range(len(na)):
        row = {'matching_rate': na[i], 'pairs': "na", 'k': k}
        # Append the new row to the DataFrame
        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
    nn_rates.append(nn)
    na_rates.append(na)


fig,ax = plt.subplots(1,1,figsize=(15,10))

sns.set_style("darkgrid")
sns.set_palette("pastel")
sns.swarmplot(data=df, x="k", y="matching_rate", hue="pairs", ax=ax,size=15,dodge=True)
ax.legend(["N$\leftrightarrow$N","N$\leftrightarrow$A"],fontsize=26,loc=[0.02,0.85])
ax.set_ylabel('Matching rate',fontsize=36)
ax.set_xlabel('k',fontsize=36)
ax.tick_params(axis='both', which='major', labelsize=32)
plt.tight_layout()

plt.savefig(f'{save_dir}/fig4c_matching_rate.png',dpi=300)
plt.show()
print(f"N-N k=1: {np.mean(nn_rates[0])}")
print(f"N-N k=3: {np.mean(nn_rates[1])}")
print(f"N-N k=5: {np.mean(nn_rates[2])}")
print(f"N-A k=1: {np.mean(na_rates[0])}")
print(f"N-A k=3: {np.mean(na_rates[1])}")
print(f"N-A k=5: {np.mean(na_rates[2])}")
# %% Fig 4.d (GWD swarm plot)

# get the minimum GWDs for neurotypical-atypical alignment
na_gwd_list = []
for i in range(5):
    result = np.load(os.path.join(data_dir,f"optimization_results_Group{i}_Group5_n=5_ntrials=10_epsilon=0.02to0.2_n_eps=20.npy"),allow_pickle=True)
    gwd_min,_ = get_min_gwd(result)
    na_gwd_list.append(gwd_min)

# get the minimum GWDs for neurotypical-neurotypical alignment
nn_gwd_list = []
for i in range(5):
    for j in range(i+1,5):
        result = np.load(os.path.join(data_dir,f"optimization_results_Group{i}_Group{j}_n=5_ntrials=10_epsilon=0.02to0.2_n_eps=20.npy"),allow_pickle=True)
        gwd_min,_ = get_min_gwd(result)
        nn_gwd_list.append(gwd_min)

nn_gwd_list = np.asarray(nn_gwd_list)
na_gwd_list = np.asarray(na_gwd_list)
gwd_list = [nn_gwd_list,na_gwd_list]


fig,ax = plt.subplots(1,1,figsize=(13,10))
sns.set_style("darkgrid")
sns.set_palette("pastel")
sns.swarmplot(data=gwd_list,ax=ax,size=25)
ax.set_ylabel("GWD",fontsize=36)
ax.set_xticks([0,1],["N$\leftrightarrow$N","N$\leftrightarrow$A"],fontsize=32)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_title("Comparison of GWD",fontsize=30)
plt.tight_layout()
plt.savefig(f'{save_dir}/fig4d_gwd_swarm_plot.png',dpi=300)
plt.show()
# %% Fig 4. e (MDS embeddings)

# color-neurotypical MDS
fig_size = (15,15)
marker = "o"
view_init = (15,150)
save_name = "fig4e_neurotypical_mds.png"
plot_MDS_embedding(embedding_list[0],marker,new_color_order,alpha=1,s=300,\
                   fig_size=(15,15),save=True,fig_dir=save_dir,save_name=save_name,view_init=view_init)

# color-atypical MDS
marker = "X"
save_name = "fig4e_atypical_mds.png"
plot_MDS_embedding(embedding_list[5],marker,new_color_order,alpha=1,s=300,\
                   fig_size=(15,15),save=True,fig_dir=save_dir,save_name=save_name,view_init=view_init)


# %% Calculate correlations between dissimilarity matrices

corr_mat = np.zeros((n_groups,n_groups))
nn_corrs = [] # neurotypical-neurotypical correlations
na_corrs = [] # neurotypical-atypical correlations
triu_indices = np.triu_indices(all_matrices[0].shape[0], k=1)
for i in range(n_groups):
    for j in range(i+1,n_groups):
        ut_i = all_matrices[i][triu_indices]
        ut_j = all_matrices[j][triu_indices]

        # Calculate the correlation
        corr, _ = pearsonr(ut_i, ut_j)
        corr_mat[i,j]  = corr
        if i == 5 or j==5:
            na_corrs.append(corr)
        else:
            nn_corrs.append(corr)

print(f'average correlation N-N: {np.mean(nn_corrs)}')
print(f'average correlation N-A: {np.mean(na_corrs)}')

