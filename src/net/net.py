import abc
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, average_precision_score


from netgan import utils
from net.utils import scores_matrix_from_transition_matrix, update_dict_of_lists, get_plot_grid_size, y_fmt, \
                      argmax_with_patience

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
        self.EO_criterion = EO_criterion

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


class OverlapStopper(Logger):
    def __init__(self, train_graph, mixing_coeff=1.0, EO_criterion=0.5, test_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.mixing_coeff = mixing_coeff
        self.test_every = test_every
        self.EO_criterion = EO_criterion
        self.stop = False

    def log(self, step, loss, x, logits, labels, metrics, model):
        stop = False
        if step % self.test_every == self.test_every-1:
            transition_matrix = model(torch.arange(start=0, end=self.train_graph.shape[0], dtype=int))
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True,
                                                                 mixing_coeff=self.mixing_coeff)
            scores_matrix = sp.csr_matrix(scores_matrix)
            sampled_graph = utils.graph_from_scores(scores_matrix, self._E)
            overlap = utils.edge_overlap(self.train_graph, sampled_graph)/self._E
            print(f'Step: {step}, Loss: {loss:.5f}, Edge-Overlap: {overlap:.3f}')
            if overlap>=self.EO_criterion:
                stop = True                
        return stop
         

class GraphStatisticsLogger(Logger):
    def __init__(self, train_graph, val_ones, val_zeros, mixing_coeff=1.0, log_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.val_ones = val_ones
        self.val_zeros = val_zeros
        self.actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))
        self.mixing_coeff = mixing_coeff
        self.log_every = log_every
        self.dict_of_lists_of_statistic = {}
        self.reference_dict_of_statistics = utils.compute_graph_statistics(self.train_graph)
        self.reference_dict_of_statistics['overlap'] = 1

    def log(self, step, loss, x, logits, labels, metrics, model):
        if step % self.log_every == self.log_every-1:
            transition_matrix = model(torch.arange(start=0, end=self.train_graph.shape[0], dtype=int))
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True,
                                                                 mixing_coeff=self.mixing_coeff)
            scores_matrix = sp.csr_matrix(scores_matrix)
            sampled_graph = utils.graph_from_scores(scores_matrix, self._E)
            statistics = utils.compute_graph_statistics(sampled_graph)
            edge_scores = np.append(scores_matrix[tuple(self.val_ones.T)].A1, 
                                    scores_matrix[tuple(self.val_zeros.T)].A1)
            statistics['step'] = step
            statistics['overlap'] = utils.edge_overlap(self.train_graph, sampled_graph)/self._E
            statistics['val_performance'] = (roc_auc_score(self.actual_labels_val, edge_scores),
                                             average_precision_score(self.actual_labels_val, edge_scores))
            self.dict_of_lists_of_statistic = update_dict_of_lists(self.dict_of_lists_of_statistic, statistics)

    def print_statistics(self, keys, EO_criterion=0.52, max_patience_for_VAL=5):
        n_rows, n_cols = get_plot_grid_size(len(keys))
        f, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(12, 12))
        plt.tight_layout(pad=2)
        steps = self.dict_of_lists_of_statistic['step']
        EO_criterion = np.argmax(np.array(self.dict_of_lists_of_statistic['overlap'])>EO_criterion)
        sum_val_performances = [np.sum(performances) for performances in self.dict_of_lists_of_statistic['val_performance']]
        VAL_criterion = argmax_with_patience(sum_val_performances, max_patience=max_patience_for_VAL)
        for row in range(n_rows):
            for col in range(n_cols):
                i = row * n_cols + col
                if i < len(keys):
                    key = keys[row * n_cols + col]
                    axs[row, col].set_title(key)
                    axs[row, col].plot(steps, self.dict_of_lists_of_statistic[key])
                    axs[row, col].hlines(y=self.reference_dict_of_statistics[key],
                                         xmin=steps[0],
                                         xmax=steps[-1],
                                         colors='green',
                                         linestyles='dashed')
                    axs[row, col].axvline(x=steps[EO_criterion], color='grey', linestyle='dashdot')
                    axs[row, col].axvline(x=steps[VAL_criterion], color='red', linestyle='dashdot')         
                    axs[row, col].yaxis.set_major_formatter(FuncFormatter(y_fmt))
                else:
                    axs[row, col].axis('off')
        plt.show()



class Net(object):
    def __init__(self, N, H, affine=False, loss_fn=torch.nn.functional.cross_entropy, loggers=[], metric_fns={}, 
                 stoppers=[]):
        self.affine = affine
        self.step = 0
        self.loss_fn = loss_fn
        self.loggers = loggers
        self.stoppers = stoppers
        self.metric_fns = metric_fns
        self._optimizer = None
        self.w_down = (0.1 * torch.randn(N, H, device=device, dtype=dtype)).clone().detach().requires_grad_()
        self.w_up = (0.1 * torch.randn(H, N, device=device, dtype=dtype)).clone().detach().requires_grad_()
        self.b_up = (0. * torch.randn(N, device=device, dtype=dtype)).clone().detach().requires_grad_()
              
    def __call__(self, x):
        return torch.nn.functional.softmax(self.predict_logits(x),
                                           dim=-1).detach().numpy()
    
    def predict_logits(self, x):
        logits = self.w_down[x] @ self.w_up
        if self.affine:
            logits += self.b_up
        return logits
    
    def _train_step(self, x, labels):
        logits = self.predict_logits(x)
        loss = self.loss_fn(logits, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return logits, loss.item()
    
    def train(self, generator, steps, optimizer_fn, optimizer_args):
        self._optimizer = optimizer_fn([self.w_down, self.w_up, self.b_up] ,**optimizer_args)
        for self.step in range(self.step, steps+self.step):
            x, labels = next(generator)
            logits, loss = self._train_step(x, labels)
            metrics = {}
            for metric_name, metric_fn in self.metric_fns.items():
                metrics[metric_name] = metric_fn(x, logits, labels)
            for logger in self.loggers:
                logger.log(self.step, loss, x, logits, labels, metrics, model=self)
            for stopper in self.stoppers:
                stop = stopper.log(self.step, loss, x, logits, labels, metrics, model=self)
                if stop:
                    return
