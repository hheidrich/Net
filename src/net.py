import sys
sys.path.insert(0, '../src/')
from pathlib import Path
import os
import pickle

import abc
import time

import numpy as np
from scipy.sparse import save_npz

import torch
device = 'cpu'
dtype = torch.float32

import utils


class Callback(abc.ABC):
    def __init__(self, invoke_every):
        self.training_stopped = False
        self.invoke_every = invoke_every
        
    def __call__(self, loss, model):
        if model.step % self.invoke_every == 0:
            self.invoke(loss, model)
        
    def stop_training(self):
        self.training_stopped = True
    
    @abc.abstractmethod
    def invoke(self, loss, model):
        pass


class OverlapTracker(Callback):
    """
    This callback serves in three ways:
    - It samples a graph from the model and saves it on hard drive.
    - It tracks the EdgeOverlap and stops if the limit is met.
    - It tracks the validation AUC-ROC score and the average precision.
    - It tracks the total time.
    """
    def __init__(self, logdir=None, invoke_every=100, EO_limit=1., val_edges=(None, None)):
        super().__init__(invoke_every)
        self.logdir = logdir
        if self.logdir is None:
            self.logs = []
        self.EO_limit = EO_limit
        self.overlap_dict = {}
        self.roc_auc_dict = {}
        self.avg_prec_dict = {}
        self.time_dict = {}
        (self.val_ones, self.val_zeros) = val_edges

    def invoke(self, loss, model):
        start = time.time()
        sampled_graph = model.sample_graph()
        overlap = utils.edge_overlap(model.A_sparse, sampled_graph) / model.num_edges
        self.overlap_dict[model.step] = overlap
        overlap_time = time.time() - start
        model.total_time += overlap_time
        self.time_dict[model.step] = model.total_time
        
        if self.val_ones is not None and self.val_zeros is not None:
            roc_auc, avg_prec = utils.link_prediction_performance(model._scores_matrix,
                                                                  self.val_ones,
                                                                  self.val_zeros)
            self.roc_auc_dict[model.step] = roc_auc
            self.avg_prec_dict[model.step] = avg_prec
        
        step_str = f'{model.step:{model.step_str_len}d}'
        print(f'Step: {step_str}/{model.steps}, Loss: {loss:.5f}, Edge-Overlap: {overlap:.3f}')
        if overlap >= self.EO_limit:
            self.stop_training()
        
        if self.logdir:
            filename = f'graph_{model.step:0{model.step_str_len}d}'
            save_npz(file=os.path.join(self.logdir, filename),
                     matrix=sampled_graph)
            
            if self.training_stopped or model.step==model.steps:
                utils.save_dict(self.overlap_dict, os.path.join(self.logdir, 'overlap.pickle'))
                utils.save_dict(self.time_dict, os.path.join(self.logdir,'timing.pickle'))

                if self.val_ones is not None and self.val_zeros is not None:
                    utils.save_dict(self.roc_auc_dict, os.path.join(self.logdir,'ROC-AUC.pickle'))
                    utils.save_dict(self.avg_prec_dict, os.path.join(self.logdir,'avg_prec.pickle'))
        else:
            self.logs.append(sampled_graph)


class WeightWatcher(Callback):
    """
    Saves the model's weights on hard drive.
    """
    def __init__(self, logdir, invoke_every=100):
        super().__init__(invoke_every)
        self.logdir = logdir
        
    def invoke(self, loss, model):
        filename =  f'weights_{model.step:0{model.step_str_len}d}'
        np.savez(file=os.path.join(self.logdir, filename),
                 W_down=model.W_down.detach().numpy(),
                 W_up=model.W_up.detach().numpy())
        pass


class Net(object):
    def __init__(self, A, H, loss_fn=None, callbacks=[]):
        self.num_edges = A.sum()/2
        self.A_sparse = A
        self.A = torch.tensor(A.toarray())
        self.step = 1
        self.callbacks = callbacks
        self._optimizer = None
        
        N = A.shape[0]
        gamma = np.sqrt(2/(N+H))
        self.W_down = (gamma * torch.randn(N, H, device=device, dtype=dtype)).clone().detach().requires_grad_()
        self.W_up = (gamma * torch.randn(H, N, device=device, dtype=dtype)).clone().detach().requires_grad_()
        
        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = self.built_in_loss_fn
        
        self.total_time = 0
              
    def __call__(self):
        return torch.nn.functional.softmax(self.get_W(), dim=-1).detach().numpy()
    
    def get_W(self):
        W = torch.mm(self.W_down, self.W_up)
        W -= W.max(dim=-1, keepdims=True)[0]
        #if self.force_W_symmetric:
        #    W = torch.max(W, W.T)
        return W
    
    def built_in_loss_fn(self, W, A, num_edges):
        """
        Computes the weighted cross-entropy loss in logits with weight matrix M * P.
        Parameters
        ----------
        W: torch.tensor of shape (N, N)
                Logits of learnable (low rank) transition matrix.

        Returns
        -------
        loss: torch.tensor (float)
                Loss at logits.
        """
        d = torch.log(torch.exp(W).sum(dim=-1, keepdims=True))
        loss = .5 * torch.sum(A * (d * torch.ones_like(A) - W)) / num_edges
        return loss
    
    def _closure(self):
        W = self.get_W()
        loss = self.loss_fn(W=W, A=self.A, num_edges=self.num_edges)
        self._optimizer.zero_grad()
        loss.backward()
        return loss
        
    def _train_step(self):
        time_start = time.time()
        loss = self._optimizer.step(self._closure)
        time_end = time.time()
        return loss.item(), (time_end - time_start)
    
    def train(self, steps, optimizer_fn, optimizer_args, EO_criterion=None):
        self._optimizer = optimizer_fn([self.W_down, self.W_up], **optimizer_args)
        self.steps = steps
        self.step_str_len = len(str(steps))
        stop = False
        for self.step in range(self.step, steps+self.step):
            loss, time = self._train_step()
            self.total_time += time
            for callback in self.callbacks:
                callback(loss=loss, model=self)
                stop = stop or callback.training_stopped    
            if stop: break
                
    def sample_graph(self):
        transition_matrix = self()
        self._scores_matrix = utils.scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                         symmetric=True)
        sampled_graph = utils.graph_from_scores(self._scores_matrix, self.num_edges)
        return sampled_graph


def start_experiments(num_experiments,
                      experiment_root,
                      train_graph,
                      H,
                      optimizer,
                      optimizer_args,
                      invoke_every,
                      steps,
                      loss_fn=None,
                      val_edges=(None, None)):
    """Start multiple experiments."""
    # create root folder
    Path(experiment_root).mkdir(parents=True, exist_ok=True)
    netmodels = []
    for experiment in range(num_experiments):
        # create experiment folder
        path = os.path.join(experiment_root, f'Experiment_{experiment:0{len(str(num_experiments-1))}d}')
        
        path_graphs = os.path.join(path, 'sampled_graphs')
        Path(path_graphs).mkdir(parents=True, exist_ok=True)
        
        path_weights = os.path.join(path, 'weights')
        Path(path_weights).mkdir(parents=True, exist_ok=True)
        
        # initialize model
        netmodel = Net(A=train_graph,
                       H=H,
                       callbacks=[OverlapTracker(logdir=path_graphs,
                                                 invoke_every=invoke_every,
                                                 EO_limit=1.,
                                                 val_edges=val_edges),
                                  WeightWatcher(logdir=path_weights,
                                                invoke_every=invoke_every)],
                       loss_fn=loss_fn)
        
        # train model
        print(f'\nExperiment_{experiment:0{len(str(num_experiments))}d}')
        netmodel.train(steps=steps,
               optimizer_fn=optimizer,
               optimizer_args=optimizer_args)
        netmodels.append(netmodel)
    return netmodels