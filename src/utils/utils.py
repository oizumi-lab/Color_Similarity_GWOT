"""
Utility functions

Genji Kawakita
"""
import os
import numpy as np
import pandas as pd



def old_reorder_idxs(col_dict,new_order):
    reorder_idxs = []
    for key,val in col_dict.items():
        idx = new_order.index(val)
        reorder_idxs.append(idx)
    return reorder_idxs

def get_reorder_idxs(old_order,new_order):
    """
    Function to get the indices of the new order
    Parameters
    ----------
    old_order: list
        a list of the original order
    new_order: list
        a list of the new order
    Returns
    -------
    reorder_idxs: list
        a list of indices of the new order
    """
    reorder_idxs = []
    for color in new_order:
        idx = old_order.index(color)
        reorder_idxs.append(idx)
    return reorder_idxs

def rearrange_diss_mat(diss_mat, reorder_idxs):
    """
    Function to rearrange the dissimilarity matrix
    Parameters
    ----------
    diss_mat: numpy array
        a dissimilarity matrix (93 x 93)
    reorder_idxs: list
        a list of indices of the new order
    Returns
    -------
    diss_mat: numpy array
        a rearranged dissimilarity matrix (93 x 93)
    """
    diss_mat = diss_mat[reorder_idxs, :]
    diss_mat = diss_mat[:, reorder_idxs]
    return diss_mat

def compute_distance_matrix(mtx1,mtx2):
    N = mtx1.shape[0]
    dist_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            dist_mat[i,j] = np.linalg.norm(mtx1[i,:]-mtx2[j,:])
    return dist_mat

def get_min_gwd(out):
    """
    Get the minimum gwd and the trial number
    Parameters
    ----------
    out: list
        a list of dictionaries for optimization results
    Returns
    -------
    min_gwd: float
        the minimum gwd
    gwd_min_trial: int
        the trial number of the minimum gwd
    """
    gwd_min_trial = 0
    min_gwd = 100
    for i in range(len(out)):
        temp_min = np.min(out[i]["gw_dist_list"])
        if min_gwd > temp_min:
            min_gwd = temp_min
            gwd_min_trial = i
    return min_gwd,gwd_min_trial

def k_nearest_colors_matching_rate(dist_mat, source_dict, target_dict, k):
    """
    Function to compute the k-nearest color matching rate.
    Parameters
    ----------
    dist_mat: numpy array
        a distance matrix (93 x 93)
    source_dict: dictionary
        a dictionary that maps the index of the source color to its hex code
    target_dict: dictionary
        a dictionary that maps the index of the target color to its hex code
    k: int
        number of nearest neighbors to consider
    Returns
    -------
    count: int
        number of source colors that have a match within the top_k nearest neighbors
    correct_rate: float
        percentage of source colors that have a match within the top_k nearest neighbors
    """
    count = 0
    for i in range(dist_mat.shape[0]):
        sorted_index_array = np.argsort(dist_mat[i, :])

        # Get the indices of the top_n nearest neighbors
        top_k_indices = sorted_index_array[:k]
        
        # Check if the source color has a match within the top_n nearest neighbors
        for index in top_k_indices:
            #print(k,len(top_k_indices))
            if source_dict[i] == target_dict[index]:
                count += 1
                break

    correct_rate = count / dist_mat.shape[0] * 100
    return count, correct_rate

def get_shifted_mat(matrix, i):
    return np.roll(matrix, i, axis=1)

def get_flipped_mat(matrix):
    return np.fliplr(matrix)

def get_eval_mat(matrix, data_select, shift=0, flip=False):
    if data_select == 'disks':
        if flip:
            matrix = get_flipped_mat(matrix)
        eval_mat = get_shifted_mat(matrix, shift)
    
    if data_select == 'LsTs':
        m, n = matrix.shape # (32, 32)
        num_Ls = num_Ts = 8
        num_blocks = m // num_Ls # 4
        
        if flip:
            matrix = get_flipped_mat(matrix)
            matrix = get_shifted_mat(matrix, num_Ls)
        
        eval_mat = []
        for i in range(num_blocks):
            block_mat = matrix[:, num_Ls*i : num_Ls*(i+1)]
            block_mat = get_shifted_mat(block_mat, shift)
            eval_mat.append(block_mat)
        
        eval_mat = np.concatenate(eval_mat, axis=1)

    return eval_mat