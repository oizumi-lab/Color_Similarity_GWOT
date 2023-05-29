"""
Preprocess data collected from participants into group-averaged dissimilarity matrices

Genji Kawakita
"""
# %% Import libraries
import numpy as np
import os
import pandas as pd
from src.preprocess_utils import *
from src.utils import *


# %% Load data
neurotypical_dir = '../data/color_neurotypical/participant_matrices'
atypical_dir = '../data/color_atypical/participant_matrices'

# extract participants data
nt_file_list = os.listdir(neurotypical_dir)
nt_sbj_list = [] # neurotypical subject list
for idx,fname in enumerate(nt_file_list):
    if "csv" in fname:
        nt_sbj_list.append(fname)

at_file_list = os.listdir(atypical_dir)
at_sbj_list = [] # neurotypical subject list
for idx,fname in enumerate(at_file_list):
    if "csv" in fname:
        at_sbj_list.append(fname)

# %% Rename the participant files
nt_renamed_dir = "../data/color_neurotypical/renamed_participant_matrices"
at_renamed_dir = "../data/color_atypical/renamed_participant_matrices"
# check if the directory exists and create if not
if not os.path.exists(nt_renamed_dir):
    os.makedirs(nt_renamed_dir)
if not os.path.exists(at_renamed_dir):
    os.makedirs(at_renamed_dir)

# get the list of renamed files
nt_sbj_dict = make_participant_idx_dict(nt_sbj_list)
at_sbj_dict = make_participant_idx_dict(at_sbj_list)

nt_renamed_files = rename_participant_files(nt_sbj_dict,neurotypical_dir,nt_renamed_dir)
at_renamed_files = rename_participant_files(at_sbj_dict,atypical_dir,at_renamed_dir)

# rename the participant files and save them as npy files in the renamed directory
nt_npy_dir = "../data/color_neurotypical/numpy_data"
at_npy_dir = "../data/color_atypical/numpy_data"

# check if the directory exists and create if not
if not os.path.exists(nt_npy_dir):
    os.makedirs(nt_npy_dir)
if not os.path.exists(at_npy_dir):
    os.makedirs(at_npy_dir)

# save the renamed files as npy files
save_renamed_files_as_npy(nt_renamed_files,nt_renamed_dir,nt_npy_dir)
save_renamed_files_as_npy(at_renamed_files,at_renamed_dir,at_npy_dir)

# %% Split the neurotypical data into groups

# load all data
nt_npy_list = os.listdir(nt_npy_dir) # get the list of npy files
nt_all_data = load_all_participant_npy(nt_npy_dir,nt_npy_list)

# reorder the colors based on Lonni's color order (new_color_order)
# load color codes
old_color_order = list(np.load('../data/hex_code/original_color_order.npy'))
new_color_order = list(np.load('../data/hex_code/new_color_order.npy'))
# get the reordering indices
reorder_idxs = get_reorder_idxs(old_color_order,new_color_order)

# split the data into groups
rand_seed = 0 # random seed for slitting participants into groups
n_groups = 5 # number of groups to split the data into
fill_val = 3.5 # fill value for the missing values
save_dir = "../results/diss_mat"
file_name = f'reordered_diss_mat_num_groups_{n_groups}_seed_{rand_seed}_fill_val_{fill_val}.pickle'
save_path = os.path.join(save_dir,file_name)
group_ave_mat,grpup_count_mat,zero_idxs = split_matrix(nt_all_data,rand_seed=rand_seed,n_groups=n_groups,fill_val=fill_val,save_mat=True,save_path=save_path,reorder=True,reorder_idxs=reorder_idxs)

# %% Create the group-averaged matrices for color-atypical participants

# load all participant data
at_npy_list = os.listdir(at_npy_dir) # get the list of npy files
at_all_data = load_all_participant_npy(at_npy_dir,at_npy_list)

# make the group-averaged matrix
save_dir = "../results/diss_mat"
file_name = f'reordered_diss_mat_color_atypical_seed_{rand_seed}_fill_val_{fill_val}.npy'
save_path = os.path.join(save_dir,file_name)
ave_mat = average_matrix(at_all_data)
ave_mat = rearrange_diss_mat(ave_mat,reorder_idxs)
np.save(save_path,ave_mat)
# %%
