"""
Run Gromov-Wasserstein optimzal transport and find optimal mappings

Genji Kawakita
"""
# %% import
import numpy as np
import os
from src.utils import *
import pickle as pkl
import multiprocessing
import ot

from src.analysis_utils import *

# %%

def run_gromov(X_rdm,Y_rdm,epsilon_range,init_mat,groups,verbose=None):
    """
    Parameters
    ----------
    Returns
    ----------

    """
    N = X_rdm.shape[0]
    M = Y_rdm.shape[0]

    # marginal uniform distributions
    p = ot.unif(M) # source ditribution
    q = ot.unif(N) # target distribution

    # shuffle the oder of matrices if necessary
    out = {}

    # gromov-wasserstein
    # align Y_rdm (source) to X_rdm (target), so C1=Y_rdm, C2=X_rdm

    gw_dist_list = []
    OT_list = []
    log_list = []
    for epsilon in epsilon_range:
        OT,log = my_entropic_gromov_wasserstein(C1=Y_rdm,C2=X_rdm,p=p,q=q,loss_fun="square_loss",epsilon=epsilon,T=init_mat,max_iter=500,log=True,verbose=verbose)
        gw_dist = log['gw_dist']
        print(f'epsilon={epsilon} gw_dist={gw_dist}')
        gw_dist_list.append(gw_dist)
        OT_list.append(OT)
        log_list.append(log)

    min_index = gw_dist_list.index(min(gw_dist_list))
    min_epsilon = epsilon_range[min_index]
    OT = OT_list[min_index]
    print(f'{groups} min_epsilon = {min_epsilon}, min_gw_dist = {min(gw_dist_list)}')
    out["groups"] = groups
    out["epsilon"] = min_epsilon
    out["OT"] = OT
    out["OT_list"] = OT_list
    out["log_list"] = log_list

    out["epsilon_range"] = epsilon_range
    out["gw_dist_list"] = gw_dist_list
    out["X_rdm"] = X_rdm
    out["Y_rdm"] = Y_rdm
    return out

# %%
def random_init_GWOT(input):
    
    group_pair, mat_pair, params = input
    mat1, mat2 = mat_pair
    si, ti = group_pair
    n_colors = mat1.shape[0]
    n_groups, n_trials = params
    epsilon_start = 0.02
    epsilon_end = 0.20
    num_epsilon = 20
    epsilon_range = np.linspace(epsilon_start,epsilon_end,num_epsilon)
    save_dir = "../results/optimization/new"
    filename = os.path.join(save_dir,f"optimization_results_Group{si}_Group{ti}_n={n_groups}_ntrials={n_trials}_epsilon={epsilon_start}to{epsilon_end}_n_eps={num_epsilon}.npy")
    if os.path.exists(filename):
        print("file exsits")
        return []
    else:
    # random initialization
        init_gw_list = np.zeros((n_trials,n_colors,n_colors))
        for trial in range(n_trials):
            init_gw_list[trial,:,:] = make_random_initplan(n_colors)


        pair_out = []
        for trial in range(n_trials):
            print(f'Group{si} & Group{ti} trial #{trial+1}')
            out = run_gromov(mat1,mat2,epsilon_range=epsilon_range,init_mat=init_gw_list[trial,:,:],groups=(si,ti),verbose=None)
            pair_out.append(out)
        
        np.save(filename,pair_out) 
        return pair_out

# %%
def main():
    # load
    n_colors = 93
    n_trials = 10
    n_groups = 5
    params = (n_groups, n_trials)
    data_path = f"../results/diss_mat/reordered_diss_mat_num_groups_{n_groups}_seed_0_fill_val_3.5.pickle"
    with open(data_path, "rb") as f:
            # Write the dictionary to the file
            data = pkl.load(f)
    col_norm_mat = data["group_ave_mat"]

    col_ano_path =  "../results/diss_mat/reordered_diss_mat_color_attypical_seed_0_fill_val_3.5.npy"
    col_ano_mat = np.load(col_ano_path,allow_pickle=True)
    #col_ano_mat = col_ano_data["group_ave_mat"]
    col_ano_mat = col_ano_mat.reshape((1,n_colors,n_colors))
    group_ave_mat = np.concatenate([col_norm_mat, col_ano_mat], axis=0) 

    inputs = []
    # parallel wrt groups
    for si in range(n_groups+1):
        for ti in range(si+1, n_groups+1):
            mat1 = group_ave_mat[si,:,:]
            mat2 = group_ave_mat[ti,:,:]
            mat_pair = (mat1,mat2)
            group_pair = (si,ti)
            input = (group_pair, mat_pair, params)
            inputs.append(input)
    print("multiprocessing started...")
    with multiprocessing.Pool(processes=5) as pool:
        out_list = pool.map(random_init_GWOT, inputs)

        # pool.map(run, range(5))
    save_dir = "../results/optimization/new"
    np.save(os.path.join(save_dir,f"optimization_results_Group_n={n_groups}_ntrials={n_trials}.npy"),out_list)

if __name__ == "__main__":
    main()

