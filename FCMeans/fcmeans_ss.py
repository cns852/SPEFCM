"""
fcmeans.py : Fuzzy C-means clustering algorithm.
"""
import numpy as np
from scipy.spatial.distance import cdist

def _distance(data, centers):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.

    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers).T

def init_weight(data_c):
    weight = np.ones((len(data_c), 1))
    return weight

def update_weight(u, w):
    weight = u.dot(w)
    return weight

def _wfcmeans0(data, u_old, c, m, w):
    
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(w*data) / um.dot(w)

    d = _distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d

def wfcmeans(data, c, m, error, maxiter, w, init=None, seed=None):

    # Setup cntr
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        S = data.shape[0]
        cntr = np.random.rand(S, c)        
        init = cntr.copy()
            
    cntr = init
        
    # Setup u
    d = _distance(data.T, cntr.T)
    d = np.fmax(d, np.finfo(np.float64).eps)
    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))
    
    
    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        upre = u.copy()
        [cntr, u, Jjm, d] = _wfcmeans0(data, upre, c, m, w)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - upre) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - upre)

    return cntr, u


def spfcmeans(data, c, m, error, maxiter, chunk):
    data_chunk = np.array_split(data, chunk)
    for i in range(chunk):
        if i==0:
            w = init_weight(data_chunk[i])
            cntr, u = wfcmeans(data_chunk[i].T, c, m, error, maxiter, w, init=None)
            w_cntr = update_weight(u, w)
            dataset_u = u.copy()
        else:
            w_chunk = init_weight(data_chunk[i])
            data_chunk[i] = np.append(data_chunk[i], cntr, axis=0)
            w = np.append(w_chunk, w_cntr, axis=0)
            cntr, u = wfcmeans(data_chunk[i].T, c, m, error, maxiter, w, init=None)
            w_cntr = update_weight(u, w)
            dataset_u = np.append(dataset_u, u, axis=1)
            
    return cntr, dataset_u



