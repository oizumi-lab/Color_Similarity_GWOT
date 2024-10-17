"""
Make Fig. 1C (disagreement matrix & 3d MDS embedding) 

Genji Kawakita
"""
# %% Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from src.preprocess_utils import *
from src.utils import *
from src.plot_utils import *


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
# %% Draw MDS embedding for color-neurotypical participants
embedding = MDS(n_components=3,dissimilarity='precomputed', random_state=0)
G_3d = embedding.fit_transform(ave_mat)

fig= plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(xs=G_3d[:,0], ys=G_3d[:,1], zs=G_3d[:,2],
        marker="o", color=new_color_order, s=250, alpha=1)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_zaxis().set_visible(False)
ax.xaxis.pane.set_edgecolor('k')
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')
ax.set_facecolor('white')
plt.tight_layout()
#plt.subplots_adjust(left=0.0, right=1, bottom=0.0, top=0.8)
plt.savefig("../results/figs/fig1c_mds.png",dpi=300,transparent=True)
plt.show()
# %%
