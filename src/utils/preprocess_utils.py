"""
Utility functions for data preprocessing

Genji Kawakita
"""
import numpy as np
import random
import shutil
import os
import pandas as pd
import pickle as pkl
from itertools import groupby
from src.utils import *

def load_sim_mat(file_path):
    """
    Load dissimilarity judgment matrix from a given path

    Parameters
    ----------
    file_path: str
        path of a dissimilarity matrix
    Returns
    -------
    mat: dissimilarity matrix
    col_dict: dictionary of color labels {index: HEX code}
    """
    data = pd.read_csv(file_path)
    mat = data.values
    columns = data.columns
    colour_codes = columns.values
    col_dict = make_color_idx_dict(colour_codes)
    return mat, col_dict

def make_color_idx_dict(columns):
    """
    Function for making a dictionary that stores the index and coresponding colours
    """
    col_dict = {}
    for i,col in enumerate(columns):
        col_dict[i] = col
    return col_dict

def make_participant_idx_dict(participant_list):
    """
    Make a dictionary of participant index and filename

    Parameters
    ----------
    participant_list: list
        list of filenames of participant data (csv files)
    Returns
    -------
    participant_dict: dictionary of participant index and filenames {index: file name}
    """
    participant_list.sort() # needs to sort to make the result consistent
    participant_dict = {}
    for i,participant in enumerate(participant_list):
        participant_dict[i] = participant
    return participant_dict

def rename_participant_files(participant_dict,data_dir,save_dir):
    """
    Rename and save participant data with the indices obtained
    from make_participant_idx_dict

    Parameters
    ----------
    participant_dict: dict
        dictionary of participant index and filenames {index: file name}
    data_dir: str
        path of directory where the original participant files are located
    save_dir: str
        path of directory to save renamed files
    Returns
    -------
    None
    """
    for key,val in participant_dict.items():
        original_path = os.path.join(data_dir,val) # path of original filename
        renamed_filename = f'participant_{key}.csv' # renamed csv filename
        renamed_path = os.path.join(save_dir,renamed_filename) # path to save a renamed file
        shutil.copy(original_path,renamed_path)
    renamed_files = os.listdir(save_dir)
    renamed_files.sort()
    return renamed_files


def load_idv_mat(file_path):
    """
    Load individual dissimilarity judgment matrix from a given path

    Parameters
    ----------
    file_path: str
        path of a dissimilarity matrix
    Returns
    -------
    mat: dissimilarity matrix of one participant (not responded entries are set to nan)
    col_dict: dictionary of color labels {index: HEX code}
    """
    df = pd.read_csv(file_path,header=None)
    df.drop([0],inplace=True) # drop 1st row, which is HEX code
    colors = df.iloc[:,0]
    col_dict = make_color_idx_dict(colors)
    df.drop([0],axis=1,inplace=True) # drop 1st column, which is HEX code
    mat = df.values
    return mat, col_dict

def make_participant_filename_list(n_sbj):
    """
    Make a list of participant filenames in the order of participant indices

    Parameters
    ----------
    n_sbj: int
        number of participants
    Returns
    -------
    file_list: list
        list of participants filenames
    """
    file_list = []
    for i in range(n_sbj):
        file_list.append(f'participant_{i}.npy')
    return file_list

def load_all_participant_npy(dir_path,file_list):
    """
    Load all the participants' data and return a concaentated data

    Parameters
    ----------
    dir_path: str
        path to the directory of participant npy files
    file_list: list
        list of participants filenames
    Returns
    -------
    all_data: numpy array
        the dissimilarity rating data of all the subjects (n_sbj x 93 x 93)
    """
    n_sbj = len(file_list)

    all_data = np.zeros((n_sbj,93,93))
    for i,file in enumerate(file_list):
        mat = np.load(f'{dir_path}/{file}',allow_pickle=True)
        all_data[i,:,:] = mat

    return all_data

def save_renamed_files_as_npy(file_list,dir_path,npy_dir):
    """"
    Save all the participants' data as npy files
    Parameters
    ----------
    file_list: list
        list of participants filenames
    dir_path: str
        path to the directory of participant npy files
    npy_dir: str
        path to the directory to save npy files
    Returns
    -------
    None
    """
    for renamed_file in file_list:
        file_path = os.path.join(dir_path,renamed_file)
        prefix = renamed_file.split(".")[0]
        mat,_ = load_idv_mat(file_path)
        save_path = f'{npy_dir}/{prefix}.npy'
        np.save(save_path,mat,allow_pickle=True)

def split_matrix(all_data,rand_seed,n_groups,fill_val=3.5,save_mat=False,save_path=None,reorder=False,reorder_idxs=None):
    """
    Split all the rating data into two group averaged matrices

    Parameters
    ----------
    all_data: numpy array
        the dissimilarity rating data of all the subjects (n_sbj x 93 x 93)
    rand_seed: int
        rand_seed
    n_groups: int
        the number of groups to be splitted
    fill_val: float
        the value to fill the empty entries
    save_mat: bool
        whether to save the splitted matrices
    save_path: str
        path to save the splitted matrices
    reorder: bool
        whether to reorder the colors
    Returns
    -------
    group_avera_mat: numpy array
        a group averaged dissimilarity matrix for group 1 (93 x 93)
    group2_averaged_mat: numpy array
        a group averaged dissimilarity matrix for group 2 (93 x 93)
    group1_count_mat: numpy array
        a matrix that counts the number of responses for each entry (93 x 93)
    group2_count_mat:
        a matrix that counts the number of responses for each entry (93 x 93)
    group1_idxs: list
        a list of indices for group 1
    group2_idxs: list
        a list of indices for group 2
    """
    n_sbj = all_data.shape[0]
    n_colors = all_data.shape[1]
    group_mat = np.zeros((n_groups,n_colors,n_colors))
    group_ave_mat = np.zeros((n_groups,n_colors,n_colors))
    #group_filled_ave_mat = np.zeros((n_groups,n_colors,n_colors)) # fill nan elements by fill_val
    group_count_mat = np.zeros((n_groups,n_colors,n_colors))
    
    random.seed(rand_seed)
    # split indices into n groups
    all_idxs = list(range(n_sbj))
    random.shuffle(all_idxs)
    groups = [list(g) for k, g in groupby(enumerate(all_idxs), lambda i_x:i_x[0] // (len(all_idxs) / n_groups))]
    group_idxs = []
    for i,group in enumerate(groups):
        temp_list = []
        for idx in group:
            temp_list.append(idx[1])
        group_idxs.append(temp_list)
        print(f"The number of subjects in Group {i+1}: {len(temp_list)}")
    
    # fill in elements of group matrices
    for group_idx in range(n_groups):
        sbj_idxs = group_idxs[group_idx]
        for sbj_idx in sbj_idxs:
            group_mat[group_idx,:,:] += np.nan_to_num(all_data[sbj_idx,:,:]) 
            group_count_mat[group_idx,:,:] += ~pd.isna(all_data[sbj_idx,:,:])
    
    # check the missing elements
    zero_idxs = []
    n_missing = np.count_nonzero(group_count_mat==0)
    miss_perc = n_missing/(n_colors*n_colors*n_groups) * 100
    print(f"Number of missing elements: {n_missing} ({miss_perc:.2f}% missing)")
    for group_idx in range(n_groups):
        for i in range(n_colors):
            for j in range(n_colors):
                if group_count_mat[group_idx,i,j] < 1:
                    group_mat[group_idx,i,j] = fill_val
                    group_count_mat[group_idx,i,j] = 1 #prevents division by 0
                    zero_idxs.append((group_idx,i,j))

    # get averaged matrices
    for group_idx in range(n_groups):
        group_ave_mat[group_idx,:,:] = np.divide(group_mat[group_idx,:,:],group_count_mat[group_idx,:,:])

    # reorder the colors
    if reorder:
        n_groups = group_ave_mat.shape[0]
        for i in range(n_groups):
            group_ave_mat[i,:,:] = rearrange_diss_mat(group_ave_mat[i,:,:],reorder_idxs)
            group_count_mat[i,:,:] = rearrange_diss_mat(group_count_mat[i,:,:],reorder_idxs)

    if save_mat:
        data_dict = {"group_ave_mat":group_ave_mat,"group_count_mat":group_count_mat,"zero_idxs":zero_idxs}
        with open(save_path, "wb") as f:
            # Write the dictionary to the file
            pkl.dump(data_dict, f)
        

    return group_ave_mat,group_count_mat,zero_idxs

def average_matrix(all_data,fill_val=3.5):
    """
    Calculate the average dissimilarity matrix
    Parameters
    ----------
    all_data: numpy array
        the dissimilarity rating data of all the subjects (n_sbj x 93 x 93)
    fill_val: float
        the value to fill the empty entries
    Returns
    -------
    all_ave_mat: numpy array
        the averaged dissimilarity matrix (93 x 93)
    """
    n_sbj = all_data.shape[0]
    n_colors = all_data.shape[1]

    idxs = list(range(n_sbj))
    all_mat = np.zeros((n_colors,n_colors))
    all_count_mat = np.zeros((n_colors,n_colors))

    for idx in idxs:
        all_mat += np.nan_to_num(all_data[idx,:,:]) 
        all_count_mat += ~pd.isna(all_data[idx,:,:])
    
    for i in range(n_colors):
        for j in range(n_colors):
            if all_count_mat[i,j] < 1:
                all_mat[i,j] = fill_val
                all_count_mat[i,j] = 1 #prevents division by 0
    
    all_ave_mat = np.divide(all_mat,all_count_mat)     

    return all_ave_mat