import abc
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, average_precision_score


# from netgan import utils
from net import utils_netgan as utils
from net.utils import scores_matrix_from_transition_matrix, update_dict_of_lists, get_plot_grid_size, y_fmt, \
                      argmax_with_patience, translate_key_for_plot

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
    def __init__(self, train_graph, print_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.print_every = print_every

    def log(self, step, loss, x, logits, labels, metrics, model):
        if step % self.print_every == self.print_every-1:
            transition_matrix = model(torch.arange(start=0, end=self.train_graph.shape[0], dtype=int))
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True)
            scores_matrix = sp.csr_matrix(scores_matrix)
            sampled_graph = utils.graph_from_scores(scores_matrix, self._E)
            overlap = utils.edge_overlap(self.train_graph, sampled_graph)/self._E
            print(f'Step: {step}, Loss: {loss:.5f}, Edge-Overlap: {overlap:.3f}')


class OverlapStopper(Logger):
    def __init__(self, train_graph, test_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.test_every = test_every

    def log(self, step, loss, x, logits, labels, metrics, model):
        overlap = 0
        if step % self.test_every == self.test_every-1:
            transition_matrix = model(torch.arange(start=0, end=self.train_graph.shape[0], dtype=int))
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True)
            scores_matrix = sp.csr_matrix(scores_matrix)
            sampled_graph = utils.graph_from_scores(scores_matrix, self._E)
            overlap = utils.edge_overlap(self.train_graph, sampled_graph)/self._E
            print(f'Step: {step}, Loss: {loss:.5f}, Edge-Overlap: {overlap:.3f}')                
        return overlap
         

class GraphStatisticsLogger(Logger):
    def __init__(self, train_graph, val_ones, val_zeros, log_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.val_ones = val_ones
        self.val_zeros = val_zeros
        self.actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))
        self.log_every = log_every
        self.dict_of_lists_of_statistic = {}
        self.reference_dict_of_statistics = utils.compute_graph_statistics(self.train_graph)
        self.reference_dict_of_statistics['overlap'] = 1

    def log(self, step, loss, x, logits, labels, metrics, model):
        if step % self.log_every == self.log_every-1:
            transition_matrix = model(torch.arange(start=0, end=self.train_graph.shape[0], dtype=int))
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True)
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
        n_cols, n_rows = get_plot_grid_size(len(keys))
        plt.rcParams.update({'font.size': 18})
        f, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(22, 12))
        axs = np.array(axs).reshape(n_rows, n_cols)
        plt.tight_layout(pad=3)
        steps = self.dict_of_lists_of_statistic['step']
        EO_criterion = np.argmax(np.array(self.dict_of_lists_of_statistic['overlap'])>EO_criterion)
        sum_val_performances = [np.sum(performances) for performances in self.dict_of_lists_of_statistic['val_performance']]
        VAL_criterion = argmax_with_patience(sum_val_performances, max_patience=max_patience_for_VAL)
        for row in range(n_rows):
            for col in range(n_cols):
                i = row * n_cols + col
                if i < len(keys):
                    key = keys[row * n_cols + col]
                    axs[row, col].plot(steps, self.dict_of_lists_of_statistic[key], color='black')
                    axs[row, col].hlines(y=self.reference_dict_of_statistics[key],
                                         xmin=steps[0],
                                         xmax=steps[-1],
                                         colors='green',
                                         linestyles='dashed')
                    axs[row, col].axvline(x=steps[EO_criterion], color='grey', linestyle='dashdot')
                    axs[row, col].axvline(x=steps[VAL_criterion], color='red', linestyle='dashdot')         
                    axs[row, col].yaxis.set_major_formatter(FuncFormatter(y_fmt))
                    axs[row, col].set_xlabel('Training iteration', labelpad=5)               
                    axs[row, col].set_ylabel(translate_key_for_plot(key), labelpad=2)
                else:
                    axs[row, col].axis('off')
#         plt.savefig('../fig/our_statistics_during_training.pdf', format='pdf')                    
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
    
    def train(self, generator, steps, optimizer_fn, optimizer_args, EO_criterion=[]):
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
                overlap = stopper.log(self.step, loss, x, logits, labels, metrics, model=self)
                if EO_criterion:
                    if overlap>EO_criterion:
                        return

################################################################################################                    
    
def weighted_logreg_loss(logits, W):
    """
    Computes the weighted cross-entropy loss in logits with weight matrix M * P.
    Parameters
    ----------
    logits: torch.tensor of shape (N, N)
            Logits of learnable (low rank) transition matrix.
    W: torch.tensor of shape (N, N)
            Matrix of weight distribution.

    Returns
    -------
    loss: torch.tensor (float)
            Loss at logits.
    """    
    d = torch.log(torch.exp(logits).sum(axis=-1, keepdims=True))
    loss = torch.sum(W * (d * torch.ones_like(W) - logits))
    return loss    
                    
class NetWithoutSampling(object):
    def __init__(self, W, H, affine=False, loggers=[], stoppers=[]):
        self.W = torch.tensor(W)
        self.affine = affine
        self.step = 0
        self.loggers = loggers
        self.stoppers = stoppers
        self._optimizer = None
        N = W.shape[0]
        self.w_down = (0.1 * torch.randn(N, H, device=device, dtype=dtype)).clone().detach().requires_grad_()
        self.w_up = (0.1 * torch.randn(H, N, device=device, dtype=dtype)).clone().detach().requires_grad_()
        self.b_up = (0. * torch.randn(N, device=device, dtype=dtype)).clone().detach().requires_grad_()
              
    def __call__(self):
        return torch.nn.functional.softmax(self.predict_logits(), dim=-1).detach().numpy()
    
    def predict_logits(self):
        assert not np.isnan(self.w_down.detach().numpy()).any(), f"Step: {self.step}"
        assert not np.isnan(self.w_up.detach().numpy()).any(), f"Step: {self.step}"
        logits = torch.mm(self.w_down, self.w_up)
        if self.affine:
            logits += self.b_up
        assert not np.isnan(logits.detach().numpy()).any(), f"Step: {self.step}"
        return logits
    
    def _train_step(self):
        logits = self.predict_logits()
        loss = weighted_logreg_loss(logits=logits, W=self.W)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return logits, loss.item()
    
    def train(self, steps, optimizer_fn, optimizer_args, EO_criterion=None):
        self._optimizer = optimizer_fn([self.w_down, self.w_up, self.b_up] ,**optimizer_args)
        for self.step in range(self.step, steps+self.step):
            logits, loss = self._train_step()
            for logger in self.loggers:
                logger.log(self.step, loss, None, logits, None, None, model=self)
            for stopper in self.stoppers:
                overlap = stopper.log(self.step, loss, None, logits, None, None, model=self)
                if EO_criterion:
                    if overlap>EO_criterion:
                        return                    


class OverlapLoggerWithoutSampling(Logger):
    def __init__(self, train_graph, print_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.print_every = print_every

    def log(self, step, loss, x, logits, labels, metrics, model):
        if step % self.print_every == self.print_every-1:
            transition_matrix = model()
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True)
            scores_matrix = sp.csr_matrix(scores_matrix)
            sampled_graph = utils.graph_from_scores(scores_matrix, self._E)
            overlap = utils.edge_overlap(self.train_graph, sampled_graph)/self._E
            print(f'Step: {step}, Loss: {loss:.5f}, Edge-Overlap: {overlap:.3f}')

class OverlapStopperWithoutSampling(Logger):
    def __init__(self, train_graph, test_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.test_every = test_every

    def log(self, step, loss, x, logits, labels, metrics, model):
        overlap = 0
        if step % self.test_every == self.test_every-1:
            transition_matrix = model()
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True)
            scores_matrix = sp.csr_matrix(scores_matrix)
            sampled_graph = utils.graph_from_scores(scores_matrix, self._E)
            overlap = utils.edge_overlap(self.train_graph, sampled_graph)/self._E
            print(f'Step: {step}, Loss: {loss:.5f}, Edge-Overlap: {overlap:.3f}')                
        return overlap            
            
class GraphStatisticsLoggerWithoutSampling(Logger):
    def __init__(self, train_graph, val_ones, val_zeros, log_every=100):
        self.train_graph = train_graph.toarray()
        self._E = train_graph.sum()
        self.val_ones = val_ones
        self.val_zeros = val_zeros
        self.actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))
        self.log_every = log_every
        self.dict_of_lists_of_statistic = {}
        self.reference_dict_of_statistics = utils.compute_graph_statistics(self.train_graph)
        self.reference_dict_of_statistics['overlap'] = 1
       
    def __call__(self):
        return self.dict_of_lists_of_statistic

    def log(self, step, loss, x, logits, labels, metrics, model):
        if step % self.log_every == self.log_every-1:
            transition_matrix = model()
            scores_matrix = scores_matrix_from_transition_matrix(transition_matrix=transition_matrix,
                                                                 symmetric=True)
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
        n_cols, n_rows = get_plot_grid_size(len(keys))
        plt.rcParams.update({'font.size': 18})
        f, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(22, 12))
        axs = np.array(axs).reshape(n_rows, n_cols)
        plt.tight_layout(pad=3)
        steps = self.dict_of_lists_of_statistic['step']
        EO_criterion = np.argmax(np.array(self.dict_of_lists_of_statistic['overlap'])>EO_criterion)
        sum_val_performances = [np.sum(performances) for performances in self.dict_of_lists_of_statistic['val_performance']]
        VAL_criterion = argmax_with_patience(sum_val_performances, max_patience=max_patience_for_VAL)
        for row in range(n_rows):
            for col in range(n_cols):
                i = row * n_cols + col
                if i < len(keys):
                    key = keys[row * n_cols + col]
                    axs[row, col].plot(steps, self.dict_of_lists_of_statistic[key], color='black')
                    axs[row, col].hlines(y=self.reference_dict_of_statistics[key],
                                         xmin=steps[0],
                                         xmax=steps[-1],
                                         colors='green',
                                         linestyles='dashed')
                    axs[row, col].axvline(x=steps[EO_criterion], color='grey', linestyle='dashdot')
                    axs[row, col].axvline(x=steps[VAL_criterion], color='red', linestyle='dashdot')         
                    axs[row, col].yaxis.set_major_formatter(FuncFormatter(y_fmt))
                    axs[row, col].set_xlabel('Training iteration', labelpad=5)               
                    axs[row, col].set_ylabel(translate_key_for_plot(key), labelpad=2)
                else:
                    axs[row, col].axis('off')
#         plt.savefig('../fig/our_statistics_during_training.pdf', format='pdf')                    
        plt.show()            