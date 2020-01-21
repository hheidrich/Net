import sys
sys.path.insert(0, '../src/')
import os
import pickle
import numpy as np
from scipy.sparse import load_npz
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import utils


class Evaluation(object):
    
    def __init__(self, experiment_root, statistic_fns):
        self.experiment_root = experiment_root
        self.statistic_fns = statistic_fns    
    
    def _load_timings(self):
        return self._load('timing')
    
    def _load_overlaps(self):
        return self._load('overlap')
    
    def _load_roc_aucs(self):
        try:
            return self._load('ROC-AUC')
        except: return None
        
    def _load_avg_precs(self):
        try:
            return self._load('avg_prec')
        except: return None

    def _load(self, name):
        
        def get_filename(idx):
            filename = os.path.join(self.experiment_root,
                                    f'Experiment_{idx:0{self.str_exp_len}d}',
                                    'sampled_graphs',
                                    f'{name}.pickle')
            return filename
        
        dicts = [utils.load_dict(get_filename(idx)) for idx in range(self.num_experiments)]
        return dicts

    def load_graph(self, experiment, step):
        graph_path = os.path.join(self.experiment_root,
                                  f'Experiment_{experiment:0{self.str_exp_len}d}',
                                  'sampled_graphs',
                                  f'graph_{step:0{self.step_len}d}.npz')
        graph = load_npz(graph_path)
        return graph
    
    def compute_statistics(self):
        # parse experiment root folder
        experiment_keys = [key for key in os.listdir(self.experiment_root) if key[:11]=='Experiment_']
        self.num_experiments = len(experiment_keys)
        self.str_exp_len = len(str(self.num_experiments-1))
        
        # load overlaps and timings
        overlaps = self._load_overlaps()
        roc_aucs = self._load_roc_aucs()
        avg_precs = self._load_avg_precs()
        timings = self._load_timings()
        
        self.steps = max(timings[0].keys())
        self.step_len = len(str(self.steps))
        step_idxs = len(timings[0].keys())
        self.invoke_every = self.steps // step_idxs
        
        statistics = {name: np.zeros([self.num_experiments,
                                      step_idxs]) for name in self.statistic_fns.keys()}
        statistics['Edge Overlap (%)'] = np.zeros([self.num_experiments, step_idxs])
        if roc_aucs is not None:
            statistics['ROC-AUC Score'] = np.zeros([self.num_experiments, step_idxs])
        if avg_precs is not None:
            statistics['Average Precision'] = np.zeros([self.num_experiments, step_idxs])
        statistics['Time (s)'] = np.zeros([self.num_experiments, step_idxs])
                    
        for step_idx, step in enumerate(range(self.invoke_every, self.steps+self.invoke_every, self.invoke_every)):
            for experiment in range(self.num_experiments):
                # load sparse graph
                graph = self.load_graph(experiment, step)
                # compute statistics
                statistics['Edge Overlap (%)'][experiment, step_idx] = overlaps[experiment][step]
                if roc_aucs is not None:
                    statistics['ROC-AUC Score'][experiment, step_idx] = roc_aucs[experiment][step]
                if avg_precs is not None:
                    statistics['Average Precision'][experiment, step_idx] = avg_precs[experiment][step]
                statistics['Time (s)'][experiment, step_idx] = timings[experiment][step]
                for name, statistic_fn in self.statistic_fns.items():
                    statistics[name][experiment, step_idx] = statistic_fn(graph)
                    
        self.statistics = statistics

    def aggregate_statistics(self, num_bins, start=0, end=1):
        # binning
        overlaps = self.statistics['Edge Overlap (%)']
        lin = np.linspace(start, end, num_bins+1)
        statistics_binned = {name:[] for name in self.statistics.keys()}
        statistics_mean = {name:np.zeros(num_bins) for name in self.statistics.keys()}
        statistics_std = {name:np.zeros(num_bins) for name in self.statistics.keys()}
        for idx, (start, end) in enumerate(zip(lin[:-1], lin[1:])):
            args = np.argwhere(np.logical_and(start<overlaps, overlaps<=end))
            for name, statistic in self.statistics.items():
                statistics_binned[name].append(statistic[args[:,0], args[:,1]])
                statistics_mean[name][idx] = statistic[args[:,0], args[:,1]].mean()
                statistics_std[name][idx] = statistic[args[:,0], args[:,1]].std()
        
        self.statistics_binned = statistics_binned
        self.statistics_mean = statistics_mean
        self.statistics_std = statistics_std
        self.mean_std = (statistics_mean, statistics_std)

    def get_specific_overlap_graph(self, target_overlap):
        overlaps = self.statistics['Edge Overlap (%)']
        args = np.argwhere(target_overlap < overlaps)
        selected_args = {}
        for (experiment, step_idx) in args:
            if experiment not in selected_args or selected_args[experiment] > step_idx:
                selected_args[experiment] = step_idx

        selected_graphs = {}
        selected_statistics = {}

        for experiment, step_idx in selected_args.items():
            step = (step_idx + 1) * self.invoke_every
            selected_graphs[experiment] = self.load_graph(experiment, step)
            selected_statistics[experiment] = {}
            for name, statistic in self.statistics.items():
                selected_statistics[experiment][name] = self.statistics[name][experiment, step_idx]
    
        return selected_graphs, selected_statistics
        

def tabular_from_statistics(EO_criterion, statistics):
    tabular_mean = {}
    tabular_std = {}
    for model_name, (statistics_mean, statistics_std) in statistics.items():
        tabular_mean[model_name] = {}
        tabular_std[model_name] = {}
        # find matching EO
        overlap = statistics_mean['Edge Overlap (%)']
        try:
            arg = np.argwhere(overlap>EO_criterion).min()
        except:
            raise Exception(f'Max Edge Overlap of {model_name} is {np.nan_to_num(overlap, -1).max():.3f}')
        for statistic_name in statistics_mean.keys():
            tabular_mean[model_name][statistic_name] = statistics_mean[statistic_name][arg]
            tabular_std[model_name][statistic_name] = statistics_std[statistic_name][arg]
    return (tabular_mean, tabular_std)


def df_from_tabular(tabular, keys=None):
    mean_dicts, std_dicts = tabular
    string_tabular = {}
    for (model_key, mean_dict) in mean_dicts.items():
        std_dict = std_dicts[model_key]
        string_tabular[model_key] = {}
        for (statistc_key, mean) in mean_dict.items():
            std = std_dict[statistc_key]
            string_tabular[model_key][statistc_key] = (f'{mean:.3f} \u00B1 {std:.3f}')
    df = pd.DataFrame(string_tabular.values(), string_tabular.keys())
    if keys is not None:
        df = df[keys]
    return df


def compute_original_statistics(original_graph, statistic_fns):
    original_statistics = {}
    for statistic, fn in statistic_fns.items():
        original_statistics[statistic] = fn(original_graph)
    original_statistics['Edge Overlap (%)'] = 1
    return original_statistics


def boxplot(statistics_binned, original_statistics, min_binsize=3, save_path=None):
    # Locate bins with sufficiently many entries and remove others 
    bin_keys = [len(_bin)>=min_binsize for _bin in statistics_binned['Edge Overlap (%)']]
    statistics = {}
    for key in statistics_binned.keys():
        statistics[key] = [arr for arr, bin_key in zip(statistics_binned[key], bin_keys) if bin_key]
    # Plot at mean edge overlap for every bin
    positions = [arr.mean() for arr in statistics['Edge Overlap (%)']]
    # Make boxplot
    keys = list(statistics.keys())
    n_cols, n_rows = utils.get_plot_grid_size(len(keys))
    plt.rcParams.update({'font.size': 18})
    f, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(22, 12))
    axs = np.array(axs).reshape(n_rows, n_cols)
    plt.tight_layout(pad=3)
    for row in range(n_rows):
        for col in range(n_cols):
            i = row * n_cols + col
            if i < len(keys):
                key = keys[row * n_cols + col]
                axs[row, col].boxplot(statistics[key], positions=positions,
                                      widths=.05,
                                      showfliers=False)
                if key in original_statistics.keys():
                    axs[row, col].hlines(y=original_statistics[key],
                                         xmin=0,
                                         xmax=1,
                                         colors='green',
                                         linestyles='dashed')        
                axs[row, col].set_xlabel('Edge Overlap (%)', labelpad=5)               
                axs[row, col].set_ylabel(key, labelpad=2)
                axs[row, col].set_xticklabels([f'{EO:.2f}'[1:] for EO in positions])
            else:
                axs[row, col].axis('off')
            axs[row, col].set_xlim(0, 1)
    if save_path:
        plt.savefig(fname=save_path)
    plt.show()   
    return
                                           

def make_comparison_df(datasets, models, statistic_fns, overlap, original_graphs):
    """ Make a table/ heatmap that compares the relative error of two models at a specified edge overlap 
    for a list of datasets and a list of statistics. Always computes error of first model minus
    error of second model.
    Parameters
    ----------
    datasets: List of strings (names of datasets)
    models: Dictionary. Keys are model names, values are +1/-1 to indicate how to merge both relative errors
    statistic_fns: Dictionary. Keys are statistic names, values are functions used to compute the statistics.
    overlap: Float. Consider first graphs of each trial that achieves this overlap.
    original_graphs: Dictionary. Keys are datasets, values are corresponding train graphs.

    Returns
    -------
    comparison_dict. Dictionary. Rows are datasets, columns are graph statistics, cells are linear combination 
                                 of relative errors with the weights from models.values().
    """
    # Create comparison dict and original statistics dict
    relative_error_dict = dict.fromkeys(statistic_fns.keys(), 0)
    comparison_dict = dict.fromkeys(datasets, relative_error_dict)
    original_statistics = dict.fromkeys(datasets, None)
    for model in models.keys():
        for dataset in datasets:
            # Check if original statistic is computed. If not, compute it
            if original_statistics[dataset] is None:
                original_statistics[dataset] = compute_original_statistics(original_graphs[dataset],
                                                                           statistic_fns)
            # Extract statistics for specified model, dataset, and edge overlap
            eval_model_dataset = Evaluation(experiment_root=f'../logs/{dataset}/{model}/',
                                            statistic_fns=statistic_fns)
            eval_model_dataset.compute_statistics()
            _, overlap_statistics = eval_model_dataset.get_specific_overlap_graph(target_overlap=overlap)
            # Compute relative error for all statistics
            for statistic in statistic_fns.keys():
                rel_error = 0
                for trial in overlap_statistics.keys():
                    rel_error += np.abs(overlap_statistics[trial][statistic] - original_statistics[dataset][statistic])
                rel_error /= len(overlap_statistics.keys()) * original_statistics[dataset][statistic]
                comparison_dict[dataset][statistic] += models[model] * np.abs(rel_error)
    df = pd.DataFrame(comparison_dict.values(), comparison_dict.keys())
    
    return df

                                           
def comparison_tabular_from_df(df, annot_size=20, xlabel_size=15, ylabel_size=15, xtick_size=10, ytick_size=10,
                               x_rotation=-45, y_rotation=0):
    ax = sns.heatmap(df, annot=True, annot_kws={"size": annot_size}, cmap='RdBu_r', center=0, linewidths=1)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=x_rotation, fontsize=xtick_size)
    plt.yticks(rotation=y_rotation, fontsize=ytick_size)
    ax.set_xlabel('Statistics', fontsize=xlabel_size)
    ax.set_ylabel('Datasets', fontsize=ylabel_size)
    return                                           