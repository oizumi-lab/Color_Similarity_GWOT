#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def check_dimensionality(embedding):
    print("Dimension : ", embedding.shape[1])
    # Distribution
    dim_mean = list()
    for i in range(embedding.shape[1]):
        dim_mean.append(np.max(np.abs(embedding[:, i])))
    plt.figure()
    plt.hist(dim_mean, bins = 30)
    plt.show()
    
    negative_values = 0
    small_dimensions = 0
    for i in range(embedding.shape[1]):
        negative_values += (embedding[:, i] < 0).sum()
        small_dimensions += all(np.abs(embedding[:, i] < 0.1))
    return negative_values, small_dimensions

#%%

data = "neutyp"
emb_dim = 200
Z = 128
N_groups = 2
N_trials = 75
N_sample = 1

embedding_list = np.load(f"../results/embeddings_pairs_list_{data}_emb={emb_dim}_Z={Z}_Ngroups={N_groups}_Ntrials={N_trials}_Nsample={N_sample}.npy")[0]

#%%
for embedding in embedding_list:
    negative_values, small_dimensions = check_dimensionality(embedding)
    print(small_dimensions)
# %%
