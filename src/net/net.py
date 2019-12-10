import abc
import numpy as np
import scipy.sparse as sp
import torch

from netgan import utils
from net.utils import scores_matrix_from_transition_matrix, update_dict_of_lists


dtype = torch.float32

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Logger(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def log(self, step, loss, x, logits, labels, metrics, model):
        pass


class BasicPrintLogger(Logger):
    def __init__(self, print_every=100):
        self.print_every = print_every
    
    def log(self, step, loss, x, logits, labels, metrics, model):
        if step % self.print_every == self.print_every-1:
                print(f'Step: {step}, Loss: {loss:.5f}')


class OverlapLogger(Logger):
    def __init__(self, train_graph, mixing_coeff=1.0, print_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.mixing_coeff = mixing_coeff
        self.print_every = print_every

    def log(self, step, loss, x, logits, labels, metrics, model):
        if step % self.print_every == self.print_every-1:
            transition_matrix = model(torch.arange(start=0, end=self.train_graph.shape[0], dtype=int))
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True,
                                                                 mixing_coeff=self.mixing_coeff)
            scores_matrix = sp.csr_matrix(scores_matrix)
            sampled_graph = utils.graph_from_scores(scores_matrix, self._E)
            overlap = utils.edge_overlap(self.train_graph, sampled_graph)/self._E
            print(f'Step: {step}, Loss: {loss:.5f}, Edge-Overlap: {overlap:.3f}')


class GraphStatisticsLogger(Logger):
    def __init__(self, train_graph, mixing_coeff=1.0, log_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.mixing_coeff = mixing_coeff
        self.log_every = log_every
        self.dict_of_lists_of_statistic = {}

    def log(self, step, loss, x, logits, labels, metrics, model):
        if step % self.log_every == self.log_every-1:
            transition_matrix = model(torch.arange(start=0, end=self.train_graph.shape[0], dtype=int))
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True,
                                                                 mixing_coeff=self.mixing_coeff)
            scores_matrix = sp.csr_matrix(scores_matrix)
            sampled_graph = utils.graph_from_scores(scores_matrix, self._E)
            statistics = utils.compute_graph_statistics(sampled_graph)
            statistics['step'] = step
            statistics['overlap'] = utils.edge_overlap(self.train_graph, sampled_graph)/self._E
            self.dict_of_lists_of_statistic = update_dict_of_lists(self.dict_of_lists_of_statistic, statistics)



class Net(object):
    def __init__(self, N, H, loss_fn=torch.nn.functional.cross_entropy, loggers=[], metric_fns={}):
        self.step = 0
        self.loss_fn = loss_fn
        self.loggers = loggers
        self.metric_fns = metric_fns
        self._optimizer = None
        self.w_down = (0.1 * torch.randn(N, H, device=device, dtype=dtype)).clone().detach().requires_grad_()
        self.w_up = (0.1 * torch.randn(H, N, device=device, dtype=dtype)).clone().detach().requires_grad_()
        self.b_up = (0. * torch.randn(N, device=device, dtype=dtype)).clone().detach().requires_grad_()
              
    def __call__(self, x):
        return torch.nn.functional.softmax(self.predict_logits(x),
                                           dim=-1).detach().numpy()
    
    def predict_logits(self, x):
        return (self.w_down[x] @ self.w_up) + self.b_up
    
    def _train_step(self, x, labels):
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return logits, loss.item()
    
    def train(self, generator, steps, optimizer_fn, optimizer_args):
        self._optimizer = optimizer_fn([self.w_down, self.w_up] ,**optimizer_args)
        for self.step in range(self.step, steps+self.step):
            x, labels = next(generator)
            logits, loss = self._train_step(x, labels)
            metrics = {}
            for metric_name, metric_fn in self.metric_fns.items():
                metrics[metric_name] = metric_fn(x, logits, labels)
            for logger in self.loggers:
                logger.log(self.step, loss, x, logits, labels, metrics, model=self)
