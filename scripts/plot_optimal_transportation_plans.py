"""
Plot optimal transportation plans between 6 groups

Genji Kawakita
"""
#%% import libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from src.utils import *
# %%
data_dir = "../results/optimization"
fig_dir = "../results/figs/ot"

max_val = 1/93
for i in range(6):
    for j in range(i+1,6):
        print(i,j)
        data = np.load(os.path.join(data_dir,f"optimization_results_Group{i}_Group{j}_n=5_ntrials=10_epsilon=0.02to0.2_n_eps=20.npy"),allow_pickle=True)
        _,gwd_min_trial = get_min_gwd(data)
        fig,ax = plt.subplots(1,1,figsize=(20,20))
        OT = data[gwd_min_trial]["OT"]
        a = sns.heatmap(OT,ax=ax,cbar=False,vmax=max_val) 
        plt.xticks([])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/OT_Group{i+1}_Group{j+1}.png',dpi=300)

# %%
