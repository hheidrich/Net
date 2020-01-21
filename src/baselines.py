import sys
sys.path.insert(0, '../src/')
import os
from pathlib import Path
import time
import abc
import numpy as np
import scipy.sparse as sp
from scipy.sparse import save_npz
import torch
import time
import pickle
import utils

class Forge(abc.ABC):
    def __init__(self, A, rank):
        self.A = A
        M = self.transform(A)
        M_LR = self._low_rank_approx(M, rank)
        A_LR = self.backtransform(M_LR, A)
        self.scores_matrix = self._normalize(A_LR)
        self.num_edges = A.sum() / 2
        
    def __call__(self, sample_size, logdir=None, val_edges=(None, None)):
        sampled_graphs = []
        for experiment in range(sample_size):
            start = time.time()
            sampled_graph = self._sample()
            sampled_graphs.append(sampled_graph)
            timing = time.time() - start
            if logdir:
                self._log(sampled_graph,
                          os.path.join(logdir,
                                       f'Experiment_{experiment:0{len(str(sample_size-1))}d}',
                                       f'sampled_graphs'),
                          timing,
                          val_edges=val_edges)
        return sampled_graphs
    
    def _low_rank_approx(self, M, rank):
        u, s, vt = sp.linalg.svds(M, k=rank, which='LM')
        M_LR = u @ np.diag(s) @ vt
        return M_LR
    
    def _normalize(self, A_LR):
        A_LR = np.maximum(A_LR, A_LR.T)
        scores_matrix = np.minimum(np.maximum(A_LR, 0), 1)
        return scores_matrix
    
    def _sample(self):
        sampled_graph = utils.graph_from_scores(self.scores_matrix, self.num_edges)
        return sampled_graph
    
    def _log(self, sampled_graph, logdir, timing, val_edges=(None, None)):
        Path(logdir).mkdir(parents=True, exist_ok=True)
        filename = 'graph_1'
        save_npz(file=os.path.join(logdir, filename),
                 matrix=sampled_graph)
        
        utils.save_dict({1: timing}, os.path.join(logdir, 'timing.pickle'))
        
        # compute overlap
        overlap = utils.edge_overlap(self.A, sampled_graph) / self.num_edges
        utils.save_dict({1: overlap}, os.path.join(logdir, 'overlap.pickle'))
        
        # evaluate link prediction performance
        val_ones, val_zeros = val_edges
        if val_ones is not None and val_zeros is not None:
            roc_auc, avg_prec = utils.link_prediction_performance(self.scores_matrix,
                                                                  val_ones,
                                                                  val_zeros)
            utils.save_dict({1: roc_auc}, os.path.join(logdir, 'ROC-AUC.pickle'))
            utils.save_dict({1: avg_prec}, os.path.join(logdir, 'avg_prec.pickle'))
    
    @abc.abstractmethod
    def transform(self, A):
        pass
    
    @abc.abstractmethod
    def backtransform(self, M_LR, A):
        pass

    
class Forge_Adjacency(Forge):
    def transform(self, A):
        return A
    
    def backtransform(self, M_LR, A):
        return M_LR
    
    
class Forge_Transition(Forge):
    def transform(self, A):
        return A.multiply(1 / A.sum(axis=-1))
    
    def backtransform(self, M_LR, A):
        M_LR = np.maximum(M_LR, 0)
        M_LR = M_LR / np.sum(M_LR, axis=-1, keepdims=True)
        scores_matrix = utils.scores_matrix_from_transition_matrix(transition_matrix=M_LR,
                                                                   symmetric=True)
        return scores_matrix
    
    def _normalize(self, A_LR):
        return A_LR
    
    
class Forge_Modularity(Forge):
    def transform(self, A):
        degrees = np.array(A.sum(axis=-1))
        KKT = degrees @ degrees.T / degrees.sum()
        M = A.toarray() - KKT
        return M
        
    def backtransform(self, M_LR, A):
        degrees = np.array(A.sum(axis=-1))
        KKT = degrees @ degrees.T / degrees.sum()
        A_LR = M_LR + KKT
        return A_LR
    
    
class Forge_SymmetricLaplacian(Forge):
    def transform(self, A):
        N = A.shape[0]
        degrees_sqrt = np.sqrt(np.array(A.sum(axis=-1)))
        degrees_sqrt_inv = 1 / degrees_sqrt
        M = sp.identity(N) - A.multiply(degrees_sqrt_inv).multiply(degrees_sqrt_inv.T)
        return M
    
    def backtransform(self, M_LR, A):
        N = A.shape[0]
        degrees = np.array(A.sum(axis=-1)).flatten()
        degrees_sqrt = np.sqrt(degrees)
        A_LR = np.diag(degrees) - degrees_sqrt * M_LR * degrees_sqrt.T
        return A_LR


def configuration_model(A, B=None, EO=None):
    """Given two graphs A and B with same amount of edges, generates new graph by keeping overlapping edges,
       and rewiring remaining edges such that degrees of nodes in A are preserved. Self-loops and multiple 
       edges are removed. If B is None, draws the percentage EO of edges from A."""
    configuration_graph = sp.csr_matrix(A.shape)
    if B is not None:
        configuration_graph = A.multiply(B)
    else:
        B = sp.csr_matrix(sp.triu(A, k=1))
        B /= B.sum()
        nonzero_ixs = B.nonzero()
        edges_from_A = np.random.choice(a=len(nonzero_ixs[0]), size=int(EO * A.sum() / 2), replace=False, 
                                        p=np.array(B[nonzero_ixs]).flatten())
        configuration_graph[nonzero_ixs[0][edges_from_A], nonzero_ixs[1][edges_from_A]] = 1
        configuration_graph = configuration_graph + configuration_graph.T
    degrees = (np.array(A.sum(axis=-1)) - np.array(configuration_graph.sum(axis=-1))).astype(int).flatten()
    stubs = np.zeros(degrees.sum())
    counter = 0
    for i in degrees.nonzero()[0]:
        stubs[counter: counter+degrees[i]] = i * np.ones(degrees[i])
        counter += degrees[i]
    np.random.shuffle(stubs)
    stubs = stubs.reshape(-1, 2).astype(int)
    configuration_graph[stubs[:, 0], stubs[:, 1]] = 1
    configuration_graph[stubs[:, 1], stubs[:, 0]] = 1  
    configuration_graph.setdiag(0)
    configuration_graph.eliminate_zeros()
    return configuration_graph


def train_and_log_multiple_conf_models(logdir,
                                       A,
                                       B=None,
                                       EOs=[0.52],
                                       experiments_per_EO=1):
    #overlaps = {str(i+1): EO for i, EO in enumerate(EOs)}
    str_len_exp = len(str(experiments_per_EO))
    str_len_step = len(str(len(EOs)-1))
    num_edges = A.sum() / 2
    for experiment in range(experiments_per_EO):
        graph_path = os.path.join(logdir,
                                  f'Experiment_{experiment:0{str_len_exp}d}',
                                  'sampled_graphs')
        
        Path(graph_path).mkdir(parents=True, exist_ok=True)
        overlaps = {}
        timings = {}
        for step, EO in enumerate(EOs):
            start = time.time()
            sampled_graph = configuration_model(A=A, B=B, EO=EO)
            overlaps[step+1] = utils.edge_overlap(A, sampled_graph)/num_edges
            timings[step+1] = time.time() - start
            
            save_npz(matrix=sampled_graph,
                     file=os.path.join(graph_path, f'graph_{step+1:0{str_len_step}d}'))
        
        utils.save_dict(overlaps, os.path.join(graph_path, 'overlap.pickle'))
        utils.save_dict(timings, os.path.join(graph_path, 'timing.pickle'))