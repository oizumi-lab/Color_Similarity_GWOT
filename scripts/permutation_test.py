#%%
import pandas as pd
from scipy import stats
import numpy as np
import itertools

#%%
def permutation_test(group1 : pd.Series, group2 : pd.Series, num_permutations):
    # オリジナルの統計値を計算
    observed_diff = group1.mean() - group2.mean()

    # 結果を保存するためのリストを作成
    permutation_results = []

    for _ in range(num_permutations):
        # データをランダムに並び替える
        combined_data = pd.concat([group1, group2])
        combined_data_permuted = combined_data.sample(frac=1, replace=False).reset_index(drop=True)

        # ランダムに並び替えたデータを2群に再分割
        permuted_group1 = combined_data_permuted[:len(group1)]
        permuted_group2 = combined_data_permuted[len(group1):]

        # ランダムに並び替えたデータから統計値を計算
        permuted_diff = permuted_group1.mean() - permuted_group2.mean()
        permutation_results.append(permuted_diff)

    # パーミュテーション結果からp値を計算
    p_value = (np.abs(permutation_results) >= np.abs(observed_diff)).mean()
    #p_value = 1 - stats.percentileofscore(np.abs(permutation_results), np.abs(observed_diff))/100

    return p_value

# %%
if __name__ == "__main__":
    ### between groups
    
    df = pd.read_csv(f"../results/accuracy_OT_top_k.csv", index_col=0)
    
    top_k_list = [1, 3, 5]
    Z_list = [20, 60, 100]
    data_list = ["N-N", "A-A", "N-A"]
    
    between_groups = list(itertools.permutations(data_list, 2))
    
    
    results_all = pd.DataFrame()
    for i, k in enumerate(top_k_list):
        
        results_for_k = pd.DataFrame()
        for j, Z in enumerate(Z_list):
            
            results_for_Z = pd.DataFrame()
            for l, (group1, group2) in enumerate(between_groups):
                results_for_pair = pd.DataFrame()

                pair = f"{group1} vs {group2}"
                
                df = df[df["top_k"] == k]
                df = df[df["Z"] == Z]
                
                df_group1 = df[df["data"] == group1]
                df_group2 = df[df["data"] == group2]
                
                p_value = permutation_test(df_group1["accuracy"], df_group2["accuracy"], num_permutations=1000)
            
                results_for_pair["pair"] = [pair]
                results_for_pair["p value"] = [p_value]
                results_for_pair["Z"] = [Z]
                results_for_pair["top k"] = [k]
                
                if l == 0:
                    results_for_Z = results_for_pair
                else:
                    results_for_Z = pd.concat([results_for_Z, results_for_pair])
            
            if j == 0:
                results_for_k = results_for_Z
            else:
                results_for_k = pd.concat([results_for_k, results_for_Z])
        
        if i == 0:
            results_all = results_for_k
        else:
            results_all = pd.concat([results_all, results_for_k])
            
    results_all.to_csv("../results/permutation_test_between_groups.csv")
    # %%
    ### vs shuffle
    df = pd.read_csv(f"../results/accuracy_OT_top_k.csv", index_col=0)
    
    results_all = pd.DataFrame()
    for i, k in enumerate(top_k_list):
        df_shuffle = pd.read_csv(f"../results/shuffle_accuracy_top{k}.csv", index_col=0)
        
        results_for_k = pd.DataFrame()
        for j, Z in enumerate(Z_list):
            
            results_for_Z = pd.DataFrame()
            for l, data in enumerate(data_list):
                results_for_data = pd.DataFrame()

                df_shuffle = df_shuffle[df_shuffle["Z"] == Z]
                df_shuffle = df_shuffle[df_shuffle["data"] == data]

                df = df[df["top_k"] == k]
                df = df[df["Z"] == Z]
                df = df[df["data"] == data]
                
                p_value = permutation_test(df["accuracy"], df_shuffle["mean accuracy"][:len(df)], num_permutations=1000)
                
                results_for_data["data"] = [data]
                results_for_data["p value"] = [p_value]
                results_for_data["Z"] = [Z]
                results_for_data["top k"] = [k]
                
                if l == 0:
                    results_for_Z = results_for_data
                else:
                    results_for_Z = pd.concat([results_for_Z, results_for_data])
            
            if j == 0:
                results_for_k = results_for_Z
            else:
                results_for_k = pd.concat([results_for_k, results_for_Z])
        
        if i == 0:
            results_all = results_for_k
        else:
            results_all = pd.concat([results_all, results_for_k])
                
    # %%
    results_all.to_csv("../results/permutation_test_vs_shuffle.csv")
# %%
