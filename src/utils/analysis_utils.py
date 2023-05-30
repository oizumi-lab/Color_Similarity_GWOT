"""
Utility functions for main analysis

Genji Kawakita
"""
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from ot.utils import list_to_array

from ot.bregman import sinkhorn
from ot.backend import get_backend
from ot.gromov import init_matrix, gwggrad,gwloss

def MDS_embedding(dist_mat,n_dim,random_state=0):
    """
    Make MDS embedding
    Parameters
    ----------
    dist_mat: numpy array
        a dissimilarity matrix (93 x 93)
    n_dim: int
        the number of dimensions of the embedding
    random_state: int
        random seed
    Returns
    -------
    embedding: numpy array
        a MDS embedding (93 x n_dim)
    """
    mds = MDS(n_components=n_dim,dissimilarity='precomputed', random_state=random_state)
    embedding = mds.fit_transform(dist_mat)
    return embedding

def procrustes_alignment(X, Y, OT):
    """
    Procrustes alignment
    Parameters
    ----------
    X: numpy array
        a matrix (n_colors x n_dim)
    Y: numpy array
        a matrix (n_colors x n_dim)
    OT: numpy array
        an optimal transportation plan (n_colors x n_colors)
    Returns
    -------
    Q: numpy array
        a rotation matrix (n_dim x n_dim)
    Y_new: numpy array
        a rotated matrix (n_colors x n_dim)
    """
    #N = X.shape[0]
    U, S, Vt = np.linalg.svd(np.matmul(Y.T, np.matmul(OT,X)))
    Q = np.matmul(U, Vt)
    Y_new = np.matmul(Y, Q)
    return Q, Y_new

"""
def procrustes_alignment(X,Y,OT):

    Procrustes alignment
    Parameters
    ----------
    X: numpy array
        a matrix (n_colors x n_dim)
    Y: numpy array
        a matrix (n_colors x n_dim)
    OT: numpy array
        an optimal transportation plan (n_colors x n_colors)
    Returns
    -------
    Q: numpy array
        a rotation matrix (n_dim x n_dim)
    Y_new: numpy array
        a rotated matrix (n_colors x n_dim)

    U, S, Vt = np.linalg.svd(np.matmul(Y.T, np.matmul(OT, X)))
    Q = np.matmul(U,Vt)
    Y_new = np.matmul(Y,Q)
    return Q, Y_new
"""

def mds_procrustes_alignment(X_rdm,Y_rdm,OT,seed=0):

    N = OT.shape[0]
    # MDS
    embedding = MDS(n_components=3,dissimilarity='precomputed', random_state=seed)
    X = embedding.fit_transform(X_rdm) # X: N x 3
    Y = embedding.fit_transform(Y_rdm) # Y: M x 3

    # Multiply transportation plan with Y (source)
    Y_OT = np.matmul(Y.T,OT*N).T # Y_OT: N x 3

    U, S, Vt = np.linalg.svd(np.matmul(X.T, Y_OT))
    P_rot = np.matmul(U,Vt)
    Y_new = np.matmul(P_rot, Y.T).T


    return Y_new


def make_random_initplan(n):
    """
    Make random initial transportation plan
    """
    # make random initial transportation plan (N x N matrix)
    T = np.random.rand(n, n) # create a random matrix of size n x n
    rep = 100 # number of repetitions
    for i in range(rep):
        # normalize each row so that the sum is 1
        p = T.sum(axis=1, keepdims=True)
        T = T / p
        # normalize each column so that the sum is 1
        q = T.sum(axis=0, keepdims=True)
        T = T / q
    T = T/n
    return T

def my_entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,T,
                                max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon(H(\mathbf{T}))

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    # add T as an input
    #T = nx.outer(p, q)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(p, q, tens, epsilon, method='sinkhorn')

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T
