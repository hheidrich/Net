import numpy as np
import torch
from scipy.sparse.linalg import eigs


def net_walker(walker):
    rw_generator = walker.walk()
    while True:
        random_walks = next(rw_generator)
        yield transform_to_transitions(random_walks)


def transform_to_transitions(random_walks):
    x = torch.tensor(random_walks[:, :-1].reshape([-1]))
    y = torch.tensor(random_walks[:, 1:].reshape([-1]))
    return x, y


def scores_matrix_from_transition_matrix(transition_matrix, mixing_coeff=1, samples=[], symmetric=True):
    """
    Compute the transition scores, i.e. the probability of a transition, for all node pairs from
    the transition matrix provided.
    Parameters
    ----------
    transition_matrix: np.array of shape (N, N)
                  The input transition matrix to count the transitions in.
    samples: integer
               If provided, scales the output to a score matrix that is comparable with the one obtained
               obtained from sampling random walks.
    mixing_coeff: float in [0,1]
               Controls contributions of stationary distribution and uniform distribution. 0 is uniform,
               1 is stationary.
    symmetric: bool, default: True
               Whether to symmetrize the resulting scores matrix.

    Returns
    -------
    scores_matrix: sparse matrix, shape (N, N)
                   Matrix whose entries (i,j) correspond to the probability of a transition from node i to j 
                   for sampling random walks from the transition matrix provided. If samples is given, corresponds
                   to the expected number of transitions.

    """
    N = transition_matrix.shape[0]
    p_stationary = np.real(eigs(transition_matrix.T, k=1, sigma=1)[1])
    p_stationary /= p_stationary.sum()
    p_marginal = mixing_coeff * p_stationary + ((1 - mixing_coeff) / N) * np.ones_like(p_stationary)
    scores_matrix = p_marginal * transition_matrix
    
    if symmetric:
        scores_matrix += scores_matrix.T
    
    if samples: 
        scores_matrix *= samples
    
    return scores_matrix


def update_dict_of_lists(dict_of_lists, new_dict):
    for key, value in new_dict.items():
        if key in dict_of_lists.keys():
            dict_of_lists[key].append(value)
        else:
            dict_of_lists[key] = [value]
    return dict_of_lists

def get_plot_grid_size(k):
    rows = int(np.ceil(np.sqrt(k)))
    if k <= rows * (rows - 1):
        cols = rows - 1
    else:
        cols = rows
    return rows, cols

def y_fmt(y, pos):    
    if np.abs(y) <= 1e-2:
        y_formatted = '{val}e-3'.format(val=int(1000 * y))
    else:
        y_formatted = round(y, 3)
    
    decades = [1e3, 1e6]
    suffix  = ["K", "M"]
    for i, d in enumerate(decades):
        if np.abs(y) >= 10 * d:
            val = y/float(d)
            y_formatted = '{val}{suffix}'.format(val=int(val), suffix=suffix[i])
    return y_formatted

def argmax_with_patience(x, max_patience):
    max_val = 0.
    patience = max_patience
    for i in range(len(x)):
        if x[i] > max_val:
            max_val = x[i]
            argmax = i
            patience = max_patience
        else:
            patience -= 1
        
        if patience == 0:
            break
    return argmax