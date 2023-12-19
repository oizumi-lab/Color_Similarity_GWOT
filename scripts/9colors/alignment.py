#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../../'))
sys.path.append(os.path.join(os.getcwd(), '../../../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
#from ...src.utils.utils import get_flipped_mat, get_shifted_mat, get_eval_mat

#%%
# list of representations where the instances of "Representation" class are included
representations = list()

# select data : "simulation", "AllenBrain", "THINGS", "DNN", "color"
regularization = True
reg_str = "reg_" if regularization else ""

data_select = f"{reg_str}9colors"

#%%
n_representations = 14 # Set the number of the instanses of "Representation". This number must be equal to or less than the number of the groups. 2 is the maximum for this data.
metric = "euclidean" # Please set the metric that can be used in "scipy.spatical.distance.cdist()".

for i in range(n_representations):#[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13]:
    name = f"Sub{i+1}" # the name of the representation
    sim_mat = np.load(f"../../data/9colors/{reg_str}similarity_matrix_sub{i+1}.npy") # the dissimilarity matrix will be computed with this embedding based on the metric
    
    representation = Representation(
        name=name,
        sim_mat=sim_mat,
        metric=metric,
        get_embedding=True, # If there is the embeddings, plese set this variable "False".
        MDS_dim=3,
    )
    
    representations.append(representation)
# %%
compute_OT = False
delete_results = False
eps_list = [0.01, 1]
device = 'cpu'
to_types = 'numpy'

# whether epsilon is sampled at log scale or not
eps_log = True

# set the number of trials, i.e., the number of epsilon values evaluated in optimization. default : 4
num_trial = 100

init_mat_plan = "random"

config = OptimizationConfig(    
    eps_list = eps_list,
    eps_log = eps_log,
    num_trial = num_trial,
    sinkhorn_method='sinkhorn', 
    
    ### Set the device ('cuda' or 'cpu') and variable type ('torch' or 'numpy')
    to_types = to_types, # user can choose "numpy" or "torch". please set "torch" if one wants to use GPU.
    device = device, # "cuda" or "cpu"; for numpy, only "cpu" can be used. 
    data_type = "double", # user can define the dtypes both for numpy and torch, "float(=float32)" or "double(=float64)". For using GPU with "sinkhorn", double is storongly recommended.
    
    ### Parallel Computation (requires n_jobs > 1, available both for numpy and torch)
    n_jobs = 3, # n_jobs : the number of worker to compute. if n_jobs = 1, normal computation will start. "Multithread" is used for Parallel computation.
    multi_gpu = True, # This parameter is only for "torch". # "True" : all the GPU installed in your environment are used, "list (e.g.[0,2,3])"" : cuda:0,2,3, and "False" : single gpu (or cpu for numpy) will use.

    db_params={"drivername": "sqlite"},
    # db_params={"drivername": "mysql+pymysql", "username": "root", "password": "****", "host": "localhost"},
    
    ### Set the parameters for optimization
    # 'uniform': uniform matrix, 'diag': diagonal matrix', random': random matrix
    init_mat_plan = init_mat_plan,
    
    n_iter = 1,
    max_iter = 1000,
    
    sampler_name = 'tpe',
    pruner_name = 'hyperband',
    pruner_params = {'n_startup_trials': 1, 
                     'n_warmup_steps': 2, 
                     'min_resource': 2, 
                     'reduction_factor' : 3
                    },
)

#%%
# Create an "AlignRepresentations" instance
align_representation = AlignRepresentations(
    config=config,
    representations_list=representations,   
   
    # histogram matching : this will adjust the histogram of target to that of source.
    histogram_matching=False,

    # metric : The metric for computing the distance between the embeddings. Please set the metric tha can be used in "scipy.spatical.distance.cdist()".
    metric="cosine", 

    # main_results_dir : folder or file name when saving the result
    main_results_dir = "../../results/gw_alignment",
   
    # data_name : Please rewrite this name if users want to use their own data.
    data_name = data_select,
    
)

#%%
sim_mat_format = "default"

visualize_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 15, 
    ot_object_tick=True,
    cmap="rocket",
    show_figure=False
)

visualize_hist = VisualizationConfig(figsize=(8, 6), color='C0')

sim_mat = align_representation.show_sim_mat(
    sim_mat_format = sim_mat_format, 
    visualization_config = visualize_config,
    visualization_config_hist = visualize_hist,
    show_distribution=False, # if True, the histogram figure of the sim_mat will be shown. visualization_config_hist will be used for adjusting this figure.
)
# %%
align_representation.RSA_get_corr(metric = "pearson", method = 'all')
# %%
visualize_config = VisualizationConfig(
    show_figure=False,
    figsize=(8, 6), 
    title_size = 15, 
    ot_object_tick=True,
    plot_eps_log=eps_log,
)

sim_mat_format = "default"
fig_dir = "../../results/figs/9colors/"

align_representation.gw_alignment(
    compute_OT = compute_OT,
    delete_results = delete_results,
    return_data = False,
    return_figure = False,
    OT_format = sim_mat_format,
    visualization_config = visualize_config,
    fig_dir=fig_dir,
    save_dataframe=True
)
# %%
align_representation.show_optimization_log(
    fig_dir=fig_dir,
    visualization_config=visualize_config
)

align_representation.calc_accuracy(
    top_k_list=[1, 2, 3],
    eval_type="ot_plan"
)

align_representation.calc_accuracy(
    top_k_list=[1, 2, 3],
    eval_type="k_nearest"
)

align_representation.plot_accuracy(
    eval_type="ot_plan",
    fig_dir=fig_dir,
    fig_name="accuracy_ot_plan.png"
)

align_representation.plot_accuracy(
    eval_type="k_nearest",
    fig_dir=fig_dir,
    fig_name="accuracy_k_nearest.png"
)

#%%
### evaluation using eval mat

def get_shifted_mat(matrix, i):
    return np.roll(matrix, i, axis=1)

def get_flipped_mat(matrix):
    return np.fliplr(matrix)

def get_eval_mat(matrix, shift=0, flip=False):
    if flip:
        matrix = get_flipped_mat(matrix)
    eval_mat = get_shifted_mat(matrix, shift)

    return eval_mat


mat_size = 9
num_shift = 9
eval_mat_org = np.eye(mat_size)
# show eval mat
plt.figure()
plt.imshow(eval_mat_org, cmap='cividis')
plt.show()

data_list = pd.DataFrame(columns=['flip', 'shift', 'matching rate', 'condition'])
for flip in [True, False]:
    for shift in range(num_shift):
        eval_mat = get_eval_mat(eval_mat_org, shift, flip)
        # show eval mat
        title = f'Eval_mat_flip={str(flip)}_shift={shift}'
        plt.figure()
        plt.imshow(eval_mat, cmap='cividis')
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title(title.replace('_', ' '))
        plt.show()
        plt.savefig(os.path.join(fig_dir, title+'.png'))
        df_acc = align_representation.calc_accuracy(top_k_list=[1], eval_type='ot_plan', eval_mat=eval_mat, return_dataframe=True)
        
        df_acc = df_acc.T
        df_acc.index.name = 'pair_name'
        # change column name
        df_acc.columns = ['matching rate']
        
        df_acc['flip'] = [str(flip)]*len(df_acc)
        df_acc['shift'] = [shift]*len(df_acc)
        df_acc['condition'] = ['all pair']*len(df_acc)
        
        data_list = pd.concat([data_list, df_acc], axis=0)

# random
for flip in [True, False]:
    for shift in range(num_shift):
        # calc accuracy for random matrices
        # shuffle OT plan
        for pairwise in align_representation.pairwise_list:
            pairwise.OT = np.random.permutation(pairwise.OT)
        
        df_random = align_representation.calc_accuracy(top_k_list=[1], eval_type='ot_plan', eval_mat=eval_mat, return_dataframe=True)

        df_random = df_random.T
        df_random.index.name = 'pair_name'
        # change column name
        df_random.columns = ['matching rate']
        
        df_random['flip'] = [str(flip)]*len(df_random)
        df_random['shift'] = [shift]*len(df_random)
        df_random['condition'] = ['random']*len(df_random)
        
        data_list = pd.concat([data_list, df_random], axis=0)
#%%
# save
save_dir = '../../results/gw_alignment/9colors/'
data_list.to_csv(os.path.join(save_dir, 'flip&shift_matching_rate.csv'))
#%%
# get the maximum top 1 for each pair
# make sure you take the maximum value of matching rate for each pair
data_list = pd.read_csv(os.path.join(save_dir, 'flip&shift_matching_rate.csv'), index_col=0)
data_list.index.name = 'pair_name'
data_list = data_list.reset_index()

# drop 'shift' and 'flip'
data_list = data_list.drop(['shift', 'flip'], axis=1)

# group by 'condition'
data_list_allpair = data_list[data_list['condition']=='all pair']
data_list_random = data_list[data_list['condition']=='random']

data_list_allpair = data_list_allpair.groupby('pair_name').max()
data_list_random = data_list_random.groupby('pair_name').max()

data_list = pd.concat([data_list_allpair, data_list_random], axis=0)

#%%
# plot using seaborn
plt.figure(figsize=(8, 6))
plt.style.use('seaborn-v0_8-darkgrid')
palette = sns.color_palette("bright", n_colors=2)

sns.swarmplot(x='condition', y='matching rate', data=data_list, palette=palette, size=8)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('matching rate')
plt.title('Matching rate for each pair')
plt.ylim([-5, 105])
plt.show()
plt.savefig(os.path.join(fig_dir, 'flip&shift_matching_rate.png'))



#%%
df_RSA_corr = align_representation.RSA_corr
np.save("../../results/gw_alignment/9colors/RSA_corr.npy", df_RSA_corr)
# %%
