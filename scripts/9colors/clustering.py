#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import manifold

#%%
RSA_corr = np.load("../../results/gw_alignment/9colors/RSA_corr.npy", allow_pickle=True).item()

corr_mat = np.zeros((14, 14))
gwd_mat = np.zeros((14, 14))
acc_mat = np.zeros((14, 14))

for i in range(14):
    for j in range(i+1, 14):
        pair_name = f"Sub{i+1}_vs_Sub{j+1}"
        
        gwd_df = pd.read_csv(f"../../results/gw_alignment/reg_9colors/reg_9colors_{pair_name}/random/reg_9colors_{pair_name}.csv")
        
        corr = RSA_corr[pair_name]
        corr_mat[i, j] = corr
        
        gwd = gwd_df["value"].min()
        gwd_mat[i, j] = gwd
        
        acc = gwd_df["user_attrs_best_acc"].iloc[gwd_df["value"].idxmin()]
        acc_mat[i, j] = acc
        
corr_mat = corr_mat + corr_mat.T + np.diag(np.ones(14))
gwd_mat = gwd_mat + gwd_mat.T + np.diag(np.zeros(14))
acc_mat = acc_mat + acc_mat.T + np.diag(np.ones(14))

# %%
plt.figure()
sns.heatmap(corr_mat)

plt.figure()
sns.heatmap(gwd_mat)

plt.figure()
sns.heatmap(acc_mat)
# %%
np.save("../../results/gw_alignment/9colors/corr_mat.npy", corr_mat)
np.save("../../results/gw_alignment/9colors/gwd_mat.npy", gwd_mat)
np.save("../../results/gw_alignment/9colors/acc_mat.npy", acc_mat)
# %%
corr_mat = 1 - corr_mat
acc_mat = 1- acc_mat
#%%
### MDS
for mat in [corr_mat, gwd_mat, acc_mat]:
    MDS_embedding = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    embedding = MDS_embedding.fit_transform(mat)
    plt.figure()
    plt.scatter(x=embedding[:, 0], y=embedding[:, 1])
    for i, (xi, yi) in enumerate(embedding):
        plt.annotate(f"{i+1}", (xi, yi))
# %%
