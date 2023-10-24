#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
#%%
data_list = ["neutyp","atyp","n-a"] #
Z_list = [16,32,64,128] # 
N_sample = 20 # number of sampling
N_trials = 75
N_seed = 20
labels = [x * N_trials for x in Z_list]
#%%
df = pd.DataFrame()
for j, data in enumerate(data_list):
    for i, Z in enumerate(Z_list):
        min_gwds = []
        df_temp = pd.DataFrame()
        for s in range(N_seed):
            gw_path = f"../results/gw_alignment/color_{data}_Z={Z}_Ntrials={N_trials}_seed{s}/color_{data}_Z={Z}_Ntrials={N_trials}_seed{s}_{data}-1_vs_{data}-2/random"
            # load .db file
            db_file = os.path.join(gw_path,f"color_{data}_Z={Z}_Ntrials={N_trials}_seed{s}_{data}-1_vs_{data}-2_random.db")
            # connect to db
            conn = sqlite3.connect(db_file)
            # display tables
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trial_values;")
            studies = cursor.fetchall()
            # find the best OT (minimum GW distance studies[i][3])
            min_gwd= 100
            best_OT_idx = 0
            for i in range(len(studies)):
                if studies[i][3] < min_gwd:
                    min_gwd = studies[i][3]
            min_gwds.append(min_gwd)
        df_temp["min_gwd"] = min_gwds
        df_temp["Z"] = [Z] * len(min_gwds)
        df_temp["data"] = [data] * len(min_gwds)
        df = pd.concat([df, df_temp])
            
name_mapping = {
    'neutyp': 'N v.s. N',
    'atyp': 'A v.s. A',
    'n-a': 'N v.s. A'
}
df['data'] = df['data'].replace(name_mapping)


#%% Correlation plot 

#%%
df = pd.DataFrame()
for j, data in enumerate(data_list):
    for i, Z in enumerate(Z_list):
        min_gwds = []
        df_temp = pd.DataFrame()
        for s in range(N_seed):
            gw_path = f"../results/gw_alignment/color_{data}_Z={Z}_Ntrials={N_trials}_seed{s}/color_{data}_Z={Z}_Ntrials={N_trials}_seed{s}_{data}-1_vs_{data}-2/random"
            # load .db file
            db_file = os.path.join(gw_path,f"color_{data}_Z={Z}_Ntrials={N_trials}_seed{s}_{data}-1_vs_{data}-2_random.db")
            # connect to db
            conn = sqlite3.connect(db_file)
            # display tables
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trial_values;")
            studies = cursor.fetchall()
            # find the best OT (minimum GW distance studies[i][3])
            min_gwd= 100
            best_OT_idx = 0
            for i in range(len(studies)):
                if studies[i][3] < min_gwd:
                    min_gwd = studies[i][3]
            min_gwds.append(min_gwd)
        df_temp["min_gwd"] = min_gwds
        df_temp["Z"] = [Z] * len(min_gwds)
        df_temp["data"] = [data] * len(min_gwds)
        df = pd.concat([df, df_temp])
            
name_mapping = {
    'neutyp': 'N v.s. N',
    'atyp': 'A v.s. A',
    'n-a': 'N v.s. A'
}
df['data'] = df['data'].replace(name_mapping)


#%%
# 色のパレットを設定する
palette = sns.color_palette("bright", n_colors=len(labels))

# エラーバー付きのプロットを作成する
plt.figure(figsize=(6, 6))
#sns.lineplot(data=df_pivot, markers=True, err_style='bars', ci='sd', linewidth=2)
#
#for i, col in enumerate(df_pivot.columns):
#    plt.errorbar(df_pivot.index, df_pivot[col], yerr=df_std[col], fmt='o', capsize=4, color=palette[i], alpha=0.5)
# make swarm plot but different columns for N-N, A-A, N-A
#sns.swarmplot(data=df_filtered, x='Z', y='accuracy', hue='data', palette='bright', size=10)
# Create a swarmplot with dodging
sns.swarmplot(data=df, x='Z', y='min_gwd', hue='data', dodge=True, palette='bright', size=3)

# Customize ticks and labels
labels = ["1200\n(Z=16)", "2400\n(Z=32)", "4800\n(Z=64)", "9600\n(Z=128)"]

plt.xticks(ticks=[0, 1, 2, 3], labels=labels)
plt.xlabel('Color combinations', size=15)
plt.ylabel('GWD', size=15)
plt.legend(title='Condition')

#plt.xticks(ticks=Z_list, labels=labels, size=25)
plt.xticks(size=15)
plt.yticks(size=15)

# 凡例を表示する
plt.legend(fontsize=15, loc='upper right')
plt.tight_layout()
plt.savefig(f"../results/figs/gwd_swarm_sampled_groups.png")

# グラフを表示する
plt.show()
# %%
