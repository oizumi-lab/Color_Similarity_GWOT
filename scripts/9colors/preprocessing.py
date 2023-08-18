#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%%

# preprocess
data_path = "../../data/9colors/"

n_sub = 14

similarity_data = pd.DataFrame(columns=["Color_1", "Color_2", "similarity"])
for i in range(n_sub):
    if i < 9:
        sub_idx = f"0{i+1}"
    else:
        sub_idx = f"{i+1}"
    
    file_name = f"color_eccen_2stim_during_{sub_idx}_MAG_1.iqdat"
    
    data_df = pd.read_csv(os.path.join(data_path, file_name), sep="\t")

    data_df = data_df[data_df["blockcode"] == "experiment"]
    data_df = data_df[data_df["Circle_1"] == -1]
    data_df = data_df[data_df["Circle_2"] == -1]
    
    similarity_data["Color_1"] = data_df["Color_1"].values
    similarity_data["Color_2"] = data_df["Color_2"].values
    similarity_data["similarity"] = data_df["similarity"].values
    
    similarity_data.to_csv(f"../../data/9colors/similarity_judgement_sub{i+1}.csv")

    # get sim mat
    similarity_matrix = np.zeros((9,9))

    for row in similarity_data.values:
        color_1 = row[0]-1
        color_2 = row[1]-1
        sim = row[2]
        similarity_matrix[color_1, color_2] = sim
    
    # make them symmetric
    similarity_matrix = (similarity_matrix + similarity_matrix.T)/2
    np.save(f"../../data/9colors/similarity_matrix_sub{i+1}.npy", similarity_matrix)
# %%
