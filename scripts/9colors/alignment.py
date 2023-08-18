#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch

from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

#%%
# list of representations where the instances of "Representation" class are included
representations = list()

# select data : "simulation", "AllenBrain", "THINGS", "DNN", "color"
data_select = "9colors"

#%%
n_representations = 14 # Set the number of the instanses of "Representation". This number must be equal to or less than the number of the groups. 2 is the maximum for this data.
metric = "euclidean" # Please set the metric that can be used in "scipy.spatical.distance.cdist()".

for i in [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13]:
    name = f"Sub{i+1}" # the name of the representation
    sim_mat = np.load(f"../../data/9colors/similarity_matrix_sub{i+1}.npy") # the dissimilarity matrix will be computed with this embedding based on the metric
    
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

align_representation.gw_alignment(
    compute_OT = compute_OT,
    delete_results = delete_results,
    return_data = False,
    return_figure = False,
    OT_format = sim_mat_format,
    visualization_config = visualize_config,
    fig_dir="../../results/figs/9colors/"
)
# %%
align_representation.show_optimization_log(
    fig_dir="../../results/figs/9colors/",
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
    fig_dir="../../results/figs/9colors/",
    fig_name="accuracy_ot_plan.png"
)

align_representation.plot_accuracy(
    eval_type="k_nearest",
    fig_dir="../../results/figs/9colors/",
    fig_name="accuracy_k_nearest.png"
)