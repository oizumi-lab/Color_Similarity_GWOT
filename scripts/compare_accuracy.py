#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
plt.style.use('seaborn-v0_8-darkgrid')

data_list = ["neutyp","atyp","n-a"]#
Z_list = [128]#16,32,64,
N_sample = 20 # number of sampling
top_k_list = [1,3,5]
N_trials = 75

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
    'neutyp': 'T v.s. T',
    'atyp': 'A v.s. A',
    'n-a': 'T v.s. A'
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

# load confidence interval data
confidence_interval = pd.read_csv("../results/confidence_interval.csv")

ci_lower_list = confidence_interval["CI Lower"].values
ci_upper_list = confidence_interval["CI Upper"].values

# CSVファイルを読み込む
df = df_OT_top_k_all #df_k_nearest_all df_OT_top_k_all

data_list = df["data"].unique()

# top_k=1の行のみを抽出する
top_k = 1
df_filtered = df[df['top_k'] == top_k]
labels = [x * N_trials for x in Z_list]

# extract Z=128
df_Z128 = df_filtered[df_filtered['Z'] == 128]
df_Z128_NN = df_Z128[df_Z128['data'] == 'N-N']
df_Z128_AA = df_Z128[df_Z128['data'] == 'A-A']
df_Z128_NA = df_Z128[df_Z128['data'] == 'N-A']

ave_NN = df_Z128_NN['accuracy'].mean()
ave_AA = df_Z128_AA['accuracy'].mean()
ave_NA = df_Z128_NA['accuracy'].mean()

print(f"ave_NN={ave_NN}, ave_AA={ave_AA}, ave_NA={ave_NA}")

palette = sns.color_palette("bright", n_colors=6)
# reverse the order of the palette
palette = palette[::-1]

plt.figure(figsize=(6, 6))
sns.swarmplot(data=df_filtered, x='Z', y='accuracy', hue='data', dodge=True, palette=palette, size=3)

# Plot chance level
plt.plot([-0.5, 3.5], [100/93, 100/93], color='black', linestyle='dashed', linewidth=2)

# Plot the std of the top-k accuracy
std = np.sqrt(100/93)
plt.fill_between([-0.5, 3.5], [ci_lower_list[0], ci_lower_list[0]], [ci_upper_list[0], ci_upper_list[0]], color='black', alpha=0.2)

labels = ["1200\n(Z=16)", "2400\n(Z=32)", "4800\n(Z=64)", "9600\n(Z=128)"]
# Customize ticks and labels
plt.xticks(ticks=[0, 1, 2, 3], labels=labels)
plt.xlabel('Color combinations', size=15)
plt.ylabel(f'Top-{top_k} matching rate', size=15)
plt.legend(title='Condition')

#plt.xticks(ticks=Z_list, labels=labels, size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(fontsize=15, loc='upper left')
plt.tight_layout()
plt.savefig(f"../results/figs/accuracy_swarm_sampled_groups_k={top_k}.png")
plt.show()

# %% swarm plot for GW distance





# %% line plot with error bar as shaded area
palette = sns.color_palette("bright", n_colors=len(labels))

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='Z', y='accuracy', hue='data', palette='bright', err_style="band", errorbar='sd', linewidth=2)
plt.xlabel('Color combinations', size=25)
plt.ylabel(f'top{top_k} accuracy', size=25)
#plt.xticks(ticks=Z_list, labels=labels, size=25)
plt.yticks(size=25)
plt.legend(title='data', fontsize=15, loc='upper left')
plt.tight_layout()
plt.savefig(f"../results/figs/accuracy_sampled_groups_k={top_k}_line_plot.png")
plt.show()
# %%

# Read the CSV file back into a DataFrame
df = pd.read_csv("../results/accuracy_OT_top_k.csv")

# Filter out only rows where Z=128
df_filtered = df[df['Z'] == 128]

# Set the Z values and top-k values you are interested in
top_k_list = [1, 3, 5]

# Create a figure and axis array (here assuming a single row of multiple columns)
fig, axs = plt.subplots(1, len(top_k_list), figsize=(18, 6))

# Iterate over each top_k value to create a subplot
for i, top_k in enumerate(top_k_list):
    # Filter for the specific top-k
    df_top_k = df_filtered[df_filtered['top_k'] == top_k]
    
    # Create the swarm plot
    sns.swarmplot(data=df_top_k, x='data', y='accuracy', dodge=True, palette='bright', size=8, ax=axs[i])
    
    # Customize ticks and labels
    axs[i].set_title(f'Top-{top_k} Matching Rate', size=20)
    axs[i].set_xlabel('Condition', size=15)
    axs[i].set_ylabel('Matching Rate', size=15)
    axs[i].tick_params(axis='both', which='major', labelsize=12)
    axs[i].legend(title='Condition', fontsize=12, loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df_OT_top_k_all is already prepared
df_filtered = df_OT_top_k_all[df_OT_top_k_all['Z'] == 128]  # Filtering only for Z=128

conditions = ['T v.s. T', 'A v.s. A', 'T v.s. A']  # The conditions you want to plot


for condition in conditions:
    plt.figure(figsize=(6, 6))

    df_condition = df_filtered[df_filtered['data'] == condition]
    
    # Creating the swarmplot
    sns.swarmplot(x='top_k', y='accuracy', data=df_condition, palette=palette, size=8)
    
    # Plot chance level on the graph
    # for k=1, 100/93, for k=3, 300/93, for k=5, 500/93
    plt.plot([-0.7, 0.5], [100/93, 100/93], color='black', linestyle='dashed', linewidth=2)
    plt.plot([0.5, 1.5], [300/93, 300/93], color='black', linestyle='dashed', linewidth=2)
    plt.plot([1.5, 2.5], [500/93, 500/93], color='black', linestyle='dashed', linewidth=2)
    
    # plot the std of the top-k accuracy
    # for k=1, 100/93, for k=3, 300/93, for k=5, 500/93
    std = np.sqrt(100/93)
    plt.fill_between([-0.7, 0.5], [ci_lower_list[0], ci_lower_list[0]], [ci_upper_list[0], ci_upper_list[0]], color='black', alpha=0.2)
    
    std = np.sqrt(300/93)
    plt.fill_between([0.5, 1.5], [ci_lower_list[1], ci_lower_list[1]], [ci_upper_list[1], ci_upper_list[1]], color='black', alpha=0.2)
    
    std = np.sqrt(500/93)
    plt.fill_between([1.5, 2.5], [ci_lower_list[2], ci_lower_list[2]], [ci_upper_list[2], ci_upper_list[2]], color='black', alpha=0.2)
    
    #plt.title(f'Top-1, 3, 5 Matching Rates for {condition} at Z=128')
    plt.xlabel('Top-k', size=35)
    plt.ylabel('Matching Rate', size=35)
    plt.xticks(ticks=[0, 1, 2], labels=['1', '3', '5'])
    plt.xticks(size=30)
    plt.yticks(size=30)
    # ylim [0,100]
    plt.ylim([-5, 105])
    plt.tight_layout()
    plt.savefig(f"../results/figs/{condition}_top_k_accuracy_swarmplot_Z128.png")
    plt.show()

# %%
