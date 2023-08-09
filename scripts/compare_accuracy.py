#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data_list = ["neutyp", "atyp", "n-a"]#
Z_list = [20, 60, 100]#
N_sample = 30 # number of sampling
top_k_list = [1, 3, 5]
N_trials = 76

df_OT_top_k_all = pd.DataFrame()
df_k_nearest_all = pd.DataFrame()

df_OT_top_k_means_all = pd.DataFrame()
df_k_nearest_means_all = pd.DataFrame()

for j, data in enumerate(data_list):
    ### for each data
    df_OT_top_k_for_Z = pd.DataFrame()
    df_k_nearest_for_Z = pd.DataFrame()
    
    df_OT_top_k_means_for_Z = pd.DataFrame()
    df_k_nearest_means_for_Z = pd.DataFrame()
    
    for i, Z in enumerate(Z_list):
        df_OT_top_k = pd.DataFrame()
        df_k_nearest = pd.DataFrame()
        
        df_OT_top_k_means = pd.DataFrame()
        df_k_nearest_means = pd.DataFrame()
        
        top_k_accuracy = pd.read_csv(f"../results/top_k_accuracy_{data}_Z={Z}_Nsample={N_sample}.csv", index_col=0).set_index("top_n")
        k_nearest_accuracy = pd.read_csv(f"../results/k_nearest_matching_rate_{data}_Z={Z}_Nsample={N_sample}.csv", index_col=0).set_index("top_n")
        
        for l, k in enumerate(top_k_list):
            df_OT_for_k = pd.DataFrame()
            df_k_nearest_for_k = pd.DataFrame()
            
            top_k_accuracy_for_k = top_k_accuracy.iloc[l, :]
            k_nearest_accuracy_for_k = k_nearest_accuracy.iloc[l, :]
            
            df_OT_for_k["accuracy"] = top_k_accuracy_for_k     
            df_OT_for_k["top_k"] = [k] * len(top_k_accuracy_for_k)
            df_OT_for_k["Z"] = [Z] * len(top_k_accuracy_for_k)
            df_OT_for_k["data"] = [data] * len(top_k_accuracy_for_k)

            df_k_nearest_for_k["accuracy"] = k_nearest_accuracy_for_k
            df_k_nearest_for_k["top_k"] = [k] * len(k_nearest_accuracy_for_k)
            df_k_nearest_for_k["Z"] = [Z] * len(k_nearest_accuracy_for_k)
            df_k_nearest_for_k["data"] = [data] * len(k_nearest_accuracy_for_k)
            
            if l == 0:
                df_OT_top_k = df_OT_for_k
                df_k_nearest = df_k_nearest_for_k
                
            else:
                df_OT_top_k = pd.concat([df_OT_top_k, df_OT_for_k])
                df_k_nearest = pd.concat([df_k_nearest, df_k_nearest_for_k])
        
        df_OT_top_k_means["top_k"] = top_k_list
        df_k_nearest_means["top_k"] = top_k_list

        df_OT_top_k_means["mean"] = top_k_accuracy.mean(axis=1).values
        df_OT_top_k_means["std"] = top_k_accuracy.std(axis=1).values
        df_OT_top_k_means["Z"] = [Z for i in range(len(top_k_list))]
        df_OT_top_k_means["data"] = [data]*len(top_k_list)
        
        df_k_nearest_means["mean"] = k_nearest_accuracy.mean(axis=1).values
        df_k_nearest_means["std"] = k_nearest_accuracy.std(axis=1).values
        df_k_nearest_means["Z"] = [Z for i in range(len(top_k_list))]
        df_k_nearest_means["data"] = [data]*len(top_k_list)
        
        if i == 0:
            df_OT_top_k_for_Z = df_OT_top_k
            df_k_nearest_for_Z = df_k_nearest
            
            df_OT_top_k_means_for_Z = df_OT_top_k_means
            df_k_nearest_means_for_Z = df_k_nearest_means
            
        else:
            df_OT_top_k_for_Z = pd.concat([df_OT_top_k_for_Z, df_OT_top_k])
            df_k_nearest_for_Z = pd.concat([df_k_nearest_for_Z, df_k_nearest])
            
            df_OT_top_k_means_for_Z = pd.concat([df_OT_top_k_means_for_Z, df_OT_top_k_means])
            df_k_nearest_means_for_Z = pd.concat([df_k_nearest_means_for_Z, df_k_nearest_means])
        
        
    #df_OT_top_k_means_for_Z["data"] = [data for i in range(len(df_OT_top_k_means_for_Z))]
    #df_k_nearest_means_for_Z["data"] = [data for i in range(len(df_k_nearest_means_for_Z))]
    
    if j == 0:
        df_OT_top_k_all = df_OT_top_k_for_Z
        df_k_nearest_all = df_k_nearest_for_Z
        
        df_OT_top_k_means_all = df_OT_top_k_means_for_Z
        df_k_nearest_means_all = df_k_nearest_means_for_Z
    
    else:
        df_OT_top_k_all = pd.concat([df_OT_top_k_all, df_OT_top_k_for_Z])
        df_k_nearest_all = pd.concat([df_k_nearest_all, df_k_nearest_for_Z])
        
        df_OT_top_k_means_all = pd.concat([df_OT_top_k_means_all, df_OT_top_k_means_for_Z])
        df_k_nearest_means_all = pd.concat([df_k_nearest_means_all, df_k_nearest_means_for_Z])

# change labels
name_mapping = {
    'neutyp': 'N-N',
    'atyp': 'A-A',
    'n-a': 'N-A'
}
df_OT_top_k_all["data"] = df_OT_top_k_all["data"].replace(name_mapping)
df_k_nearest_all["data"] = df_k_nearest_all["data"].replace(name_mapping)

df_OT_top_k_means_all["data"] = df_OT_top_k_means_all["data"].replace(name_mapping)
df_k_nearest_means_all["data"] = df_k_nearest_means_all["data"].replace(name_mapping)

# %%
df_OT_top_k_all.to_csv(f"../results/accuracy_OT_top_k.csv")
df_k_nearest_all.to_csv(f"../results/accuracy_k_nearest.csv")

df_OT_top_k_means_all.to_csv(f"../results/accuracy_OT_top_k_means.csv")
df_k_nearest_means_all.to_csv(f"../results/accuracy_k_nearest_means.csv")

# %%
# CSVファイルを読み込む
df = df_OT_top_k_all #df_k_nearest_all df_OT_top_k_all

data_list = df["data"].unique()

# top_k=1の行のみを抽出する
top_k = 1
df_filtered = df[df['top_k'] == top_k]
labels = [Z * N_trials for Z in Z_list]

# データを整形する
#df_pivot = df_filtered.pivot(index='Z', columns='data', values='mean')
#df_std = df_filtered.pivot(index='Z', columns='data', values='std')

# 色のパレットを設定する
palette = sns.color_palette("bright", n_colors=len(labels))

# エラーバー付きのプロットを作成する
plt.figure(figsize=(10, 6))
#sns.lineplot(data=df_pivot, markers=True, err_style='bars', ci='sd', linewidth=2)
#
#for i, col in enumerate(df_pivot.columns):
#    plt.errorbar(df_pivot.index, df_pivot[col], yerr=df_std[col], fmt='o', capsize=4, color=palette[i], alpha=0.5)

sns.violinplot(data=df, x='Z', y='accuracy', hue='data', palette='bright', inner='stick', scale='count')

# グラフの装飾
plt.xlabel('Color combinations', size=25)
plt.ylabel(f'top{top_k} accuracy', size=25)
#plt.xticks(ticks=Z_list, labels=labels, size=25)
plt.yticks(size=25)

# 凡例を表示する
plt.legend(title='data', fontsize=15)
plt.tight_layout()
plt.savefig(f"../results/figs/accuracy_sampled_groups_k={top_k}.png")

# グラフを表示する
plt.show()

# %%
