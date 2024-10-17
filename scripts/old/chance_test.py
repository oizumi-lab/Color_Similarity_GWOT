#%%
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from GW_methods.src.align_representations import Representation, Visualization_Config, Align_Representations, Optimization_Config
#%%
def shuffle_rows(matrix):
    for i in range(matrix.shape[0]):
        np.random.shuffle(matrix[i])
    return matrix

def calc_accuracy_with_topk_diagonal(matrix, k, order="maximum"):
        # Get the diagonal elements
        diagonal = np.diag(matrix)

        # Get the top k values for each row
        if order == "maximum":
            topk_values = np.partition(matrix, -k)[:, -k:]
        elif order == "minimum":
            topk_values = np.partition(matrix, k - 1)[:, :k]
        else:
            raise ValueError("Invalid order parameter. Must be 'maximum' or 'minimum'.")

        # Count the number of rows where the diagonal is in the top k values
        count = np.sum(np.isin(diagonal, topk_values))

        # Calculate the accuracy as the proportion of counts to the total number of rows
        accuracy = count / matrix.shape[0]
        accuracy *= 100

        return accuracy

#%%
if __name__ == "__main__":
    data_list = ["neutyp", "atyp", "n-a"]#
    Z_list = [20, 60, 100]#
    N_sample = 30 # number of sampling
    N_trials = 76

    OT_list = []

    ### get OTs
    OT_list_all = []
    for data in data_list:  
        OT_list_forZ = []
        for Z in Z_list:
            embeddings_pairs_list = np.load(f"../results/embeddings_pairs_list_{data}_Z={Z}_Ntrials={N_trials}_Nsample={N_sample}.npy")
            
            OT_list = []
            for i, embeddings_pair in enumerate(embeddings_pairs_list):
                group1 = Representation(name=f"Group1 seed={i}", embedding=embeddings_pair[0], metric="euclidean")
                group2 = Representation(name=f"Group2 seed={i}", embedding=embeddings_pair[1], metric="euclidean")

                representation_list = [group1, group2]

                opt_config = Optimization_Config(data_name=f"color {data} Z={Z} Ntrials={N_trials}", 
                                        init_plans_list=["random"],
                                        num_trial=10,
                                        n_iter=2, 
                                        max_iter=200,
                                        sampler_name="tpe", 
                                        eps_list=[0.02, 0.2],
                                        eps_log=True,
                                        )

                alignment = Align_Representations(config=opt_config, 
                                        representations_list=representation_list,
                                        metric="euclidean",
                                        )


                alignment.gw_alignment(results_dir="../results/gw alignment/",
                            load_OT=True,
                            returned="row_data",
                            show_log=False
                            )
                
                OT_list.append(alignment.pairwise_list[0].OT)
            
            OT_list_forZ.append(OT_list)
            
        OT_list_all.append(OT_list_forZ)
                
    #%%
    ### shuffle test
    top_k = 5
    trials = 10000
    
    df_all = pd.DataFrame()
    for i, data in enumerate(data_list):
        
        df_forZ = pd.DataFrame()
        for j, Z in enumerate(Z_list):
            
            df = pd.DataFrame()
            mean_accuracies = []
            for trial in tqdm(range(trials)):
                
                accuracy = 0
                for k in range(N_sample):
                    OT = OT_list_all[i][j][k]
                    shuffled_OT = shuffle_rows(OT)
                    accuracy += calc_accuracy_with_topk_diagonal(shuffled_OT, k=top_k, order="maximum")
                accuracy /= N_sample
                mean_accuracies.append(accuracy)
                
            df["mean accuracy"] = mean_accuracies
            df["Z"] = [Z] * trials
            df["data"] = [data] * trials
            
            if j == 0:
                df_forZ = df
            else:
                df_forZ = pd.concat([df_forZ, df])
        
        if i == 0:
            df_all = df_forZ
        else:
            df_all = pd.concat([df_all, df_forZ])
            
    name_mapping = {
        'neutyp': 'N-N',
        'atyp': 'A-A',
        'n-a': 'N-A'
    }
    df_all["data"] = df_all["data"].replace(name_mapping)
    
    df_all.to_csv(f"../results/shuffle_accuracy_top{top_k}.csv")
    
    #%%
    ### load data
    top_k = 1
    df_OT_all = pd.read_csv(f"../results/accuracy_OT_top_k.csv")
    df_null_all = pd.read_csv(f"../results/shuffle_accuracy_top{top_k}.csv")

    df_for_k = df_OT_all[df_OT_all['top_k'] == top_k]
    data_list = ["N-N", "A-A", "N-A"]
    
    df_p_values_all = pd.DataFrame()
    for i, data in enumerate(data_list):
        
        df_p_values_for_Z = pd.DataFrame()
        for j, Z in enumerate(Z_list):
            
            df_p_values = pd.DataFrame()

            df_for_data = df_for_k[df_for_k["data"] == data]
            df_OT = df_for_data[df_for_data["Z"] == Z]
            
            df_null_for_data = df_null_all[df_null_all["data"] == data]
            df_null = df_null_for_data[df_null_for_data["data"] == data]
            plt.hist(df_null["mean accuracy"])
            plt.show()
            
            p_value = 1 - stats.percentileofscore(df_null["mean accuracy"], df_OT["mean"])/100
            
            df_p_values["p values"] = p_value
            df_p_values["data"] = data
            df_p_values["Z"] = Z

            if j == 0:
                df_p_values_for_Z = df_p_values
            else:
                df_p_values_for_Z = pd.concat([df_p_values_for_Z, df_p_values])
            
        if i == 0:
            df_p_values_all = df_p_values_for_Z
        else:
            df_p_values_all = pd.concat([df_p_values_all, df_p_values_for_Z])
            
    df_p_values_all.to_csv(f"../results/p_values_k={top_k}.csv")
    print(df_p_values_all)
# %%
