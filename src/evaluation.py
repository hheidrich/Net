import sys
sys.path.insert(0, '../src/')
import os
import pickle
import numpy as np
from scipy.sparse import load_npz
from matplotlib import pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import utils


class Evaluation(object):
    
    def __init__(self, experiment_root, statistic_fns):
        self.experiment_root = experiment_root
        self.statistic_fns = statistic_fns
        self.compute_statistics()
    
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
    
    def get_tabular(self, experiment, step):
        step_idx = (step // self.invoke_every) - 1
        graph_stats = {name: stats[experiment, step_idx] for name, stats in self.statistics.items()}
        return graph_stats
    
    def get_seleted_average(self, experiments, steps):
        means = {}
        stds = {}
        for name, statistic in self.statistics.items():
            means[name] = statistic[experiments, steps].mean()
            stds[name] = statistic[experiments, steps].std()
        return means, stds
                                           
    def get_val_criterion(self, max_patience):
        sum_val_performances = self.statistics['ROC-AUC Score'] + self.statistics['Average Precision']
        val_steps = [utils.argmax_with_patience(x=arr, max_patience=max_patience) for arr in sum_val_performances]
        val_overlaps = [overlaps[step] for overlaps, step in zip(self.statistics['Edge Overlap (%)'], val_steps)]
        val_criterion = np.array(val_overlaps).mean()                                                                           
        return val_criterion


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


def boxplot(statistics, statistics_binned, original_statistics, min_binsize=3, max_patience_for_VAL=5, save_path=None):
    # Locate bins with sufficiently many entries and remove others 
    bin_keys = [len(_bin)>=min_binsize for _bin in statistics_binned['Edge Overlap (%)']]
    statistics = {}
    for key in statistics_binned.keys():
        statistics[key] = [arr for arr, bin_key in zip(statistics_binned[key], bin_keys) if bin_key]
    # Plot at mean edge overlap for every bin and compute VAL-criteria
    positions = [arr.mean() for arr in statistics['Edge Overlap (%)']]
    sum_val_performances = [np.sum(performances) for performances in self.dict_of_lists_of_statistic['val_performance']]
    VAL_criterion = argmax_with_patience(sum_val_performances, max_patience=max_patience_for_VAL)                              
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
#                     axs[row, col].axvline(x=steps[VAL_criterion], color='red', linestyle='dashdot')                           
                axs[row, col].set_xlabel('Edge Overlap (%)', labelpad=5)               
                axs[row, col].set_ylabel(key, labelpad=2)
                axs[row, col].set_xticklabels([f'{EO:.2f}'[1:] for EO in positions])
            else:
                axs[row, col].axis('off')
            axs[row, col].set_xlim(0, 1)
    if save_path:
        plt.savefig(fname=save_path, bbox_inches='tight')
    plt.show()   
    return
                                           

def errorbar_plot(models_statistics_binned, original_statistics, min_binsize=3, grid_size=None, figsize=(22, 12), show_keys=None, translation_dict=None, max_patience=5, plot_val=False, save_path=None):
    # Set up figure        
    if show_keys is None:
        keys = list(models_statistics_binned[list(models_statistics_binned.keys())[0]][0].keys()) 
    else:
        keys = show_keys
    translation = {}
    for key in keys:
        if translation_dict is not None and key in translation_dict.keys():
            translation[key] = translation_dict[key]
        else:
            translation[key] = key
    if grid_size:
        n_rows, n_cols = grid_size
    else:                                           
        n_cols, n_rows = utils.get_plot_grid_size(len(keys))
    plt.rcParams.update({'font.size': 22})
    f, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = np.array(axs).reshape(n_rows, n_cols)
    plt.tight_layout(pad=3)
    for counter, model in enumerate(models_statistics_binned):
        statistics_binned = models_statistics_binned[model][0]
        color = models_statistics_binned[model][1]
        if len(models_statistics_binned[model])>2:
            val_criterion = models_statistics_binned[model][2]                                              
        # Locate bins with sufficiently many entries and remove others 
        bin_keys = [len(_bin)>=min_binsize for _bin in statistics_binned['Edge Overlap (%)']]
        means, stds = {}, {}
        for key in statistics_binned.keys():
            means[key] = [arr.mean() for arr, bin_key in zip(statistics_binned[key], bin_keys) if bin_key]
            stds[key] = [arr.std() for arr, bin_key in zip(statistics_binned[key], bin_keys) if bin_key]
        # Plot at mean edge overlap for every bin
        positions = means['Edge Overlap (%)']
        # Make boxplot
        for row in range(n_rows):
            for col in range(n_cols):
                i = row * n_cols + col
                if i < len(keys):
                    key = keys[row * n_cols + col]
                    axs[row, col].errorbar(x=positions, y=means[key], yerr=stds[key], fmt=f's{color}',
                                           capsize=5, label=model)
                    if key in original_statistics.keys():
                        axs[row, col].hlines(y=original_statistics[key],
                                             xmin=0,
                                             xmax=1,
                                             colors='green',
                                             linestyles='dashed',
                                             label='Target (input graph)')  
                    if plot_val:                                           
                        axs[row, col].axvline(x=val_criterion(max_patience), color=color, linestyle='dashdot',
                                              label=f'VAL stopping-criterion ({model})')                     
                    axs[row, col].set_xlabel('Edge overlap (%)', labelpad=5)               
                    axs[row, col].set_ylabel(translation[key], labelpad=5)
                    if counter==0:                                           
                        axs[row, col].set_xticks([EO for EO in positions[::2]])                                           
                        axs[row, col].set_xticklabels([f'{EO:.2f}'[1:] for EO in positions[::2]])
#                     for tick in axs[row, col].get_xticklabels():
#                         tick.set_visible(True)  
                    if key in ['Wedge Count']:#, 'Triangle Count', 'Square Count']:
                        axs[row, col].yaxis.set_major_formatter(FuncFormatter(utils.y_fmt_K))                       
                else:
                    axs[row, col].axis('off')
                axs[row, col].set_xlim(0, max(positions)+0.05)
    handles, labels = axs[0,0].get_legend_handles_labels()
    if plot_val:                                           
        label_order = [3, 5, 1, 4, 0]      
        ncol=5                                           
    else:
        label_order = [1, 3, 2]                                           
        ncol=3                                           
    handles_sorted = [handles[i] for i in label_order]
    labels_sorted = [labels[i] for i in label_order]                                           
    f.legend(handles_sorted, labels_sorted, loc='upper center', ncol=ncol, frameon=False)
    if save_path:
        plt.savefig(fname=save_path, bbox_inches='tight')    
    plt.show()                                              
    return                                           

                                           
def make_rel_error_df(datasets, models, statistic_fns, overlap, original_graphs):
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
    comparison_dict = {}
    for dataset in datasets:
            statistic_keys = list(statistic_fns.keys())+['ROC-AUC Score', 'Average Precision']                                   
            comparison_dict[dataset] = dict.fromkeys(statistic_keys, 0)                                           
    original_statistics = dict.fromkeys(datasets, None)
    for model in models.keys():
        for dataset in datasets:
            # Check if original statistic is computed. If not, compute it
            if original_statistics[dataset] is None:
                original_statistics[dataset] = compute_original_statistics(original_graphs[dataset],
                                                                           statistic_fns)
                original_statistics[dataset]['ROC-AUC Score'] = 1
                original_statistics[dataset]['Average Precision'] = 1                                           
            # Extract statistics for specified model, dataset, and edge overlap
            eval_model_dataset = Evaluation(experiment_root=f'../logs/rel_error_table/{dataset}/{model}/',
                                            statistic_fns=statistic_fns)
            _, overlap_statistics = eval_model_dataset.get_specific_overlap_graph(target_overlap=overlap)
            # Compute relative error for all statistics
            for statistic in list(statistic_fns.keys())+['ROC-AUC Score', 'Average Precision']:                                 
                rel_error = 0
                for trial in overlap_statistics.keys():
                    rel_error += np.abs(overlap_statistics[trial][statistic] - original_statistics[dataset][statistic])
                rel_error /= len(overlap_statistics.keys()) * original_statistics[dataset][statistic]
                comparison_dict[dataset][statistic] += models[model] * np.abs(rel_error)
            # Compute average edge overlaps and print them
            avg_overlap = 0
            for trial in overlap_statistics.keys():
                avg_overlap += overlap_statistics[trial]['Edge Overlap (%)'] / len(overlap_statistics.keys())
    df = pd.DataFrame(comparison_dict.values(), comparison_dict.keys())
    return df

                                           
def heat_map_from_df(df, color_limits=None, figsize=(15, 15), annot_size=20, xlabel_size=15, ylabel_size=15, xtick_size=10, ytick_size=10, x_rotation=-45, y_rotation=0, xtick_shift=0, colorlabel_size=20, save_path=None):
    # Round values in df
    df_dict = df.to_dict()
    for outer_key, inner_dict in df_dict.items():
        for inner_key, val in inner_dict.items():
            val = round(val, 2)
            if val==0:
                val=0
            df_dict[outer_key][inner_key] = val
    df = pd.DataFrame(df_dict)
    # Make plot                                           
    f, ax = plt.subplots(figsize=figsize)                                           
    plt.tight_layout()                     
    if color_limits is not None:                                           
        ax = sns.heatmap(df, annot=True, annot_kws={"size": annot_size}, cmap='RdBu_r', linewidths=1,
                         vmin=color_limits[0],
                         vmax=color_limits[1],
                         ax=ax)
    else:
        ax = sns.heatmap(df, annot=True, annot_kws={"size": annot_size}, cmap='RdBu_r', center=0, linewidths=1)                 
    cmap = ax.figure.axes[-1].tick_params(labelsize=colorlabel_size, length=0)                                       
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    locs, labels = plt.xticks()                                           
    plt.xticks(ticks=locs-xtick_shift, rotation=x_rotation, fontsize=xtick_size)
    plt.yticks(rotation=y_rotation, fontsize=ytick_size)
    ax.set_xlabel('Statistics', fontsize=xlabel_size)
    ax.set_ylabel('Data sets', fontsize=ylabel_size) 
    ax.tick_params(axis=u'both', which=u'both',length=0)                                           
    if save_path:                                       
        plt.savefig(fname=save_path, bbox_inches='tight')                                           
    return         
                     
                                           
def df_from_dataset(path_to_dataset, statistic_fns, target_overlap, original_graph, max_trials=None):
    name_of_dataset = list(filter(None, path_to_dataset.split('/')))[-1]
    names_of_models = [x for x in os.listdir(path_to_dataset) if x[0] != '.']
    evals = {}
    means = {name_of_dataset: compute_original_statistics(original_graph, statistic_fns)}
    for name_of_model in names_of_models:
        evals[name_of_model] = Evaluation(os.path.join(path_to_dataset, name_of_model), statistic_fns)
        statistics = evals[name_of_model].get_specific_overlap_graph(target_overlap)[1]
        if max_trials is not None:
            statistics = {key:val for (key, val) in statistics.items() if key in range(max_trials)}.values()
        else:
            statistics = statistics.values()
        means[name_of_model] = {name: np.mean([elem[name] for elem in statistics]) for name in list(statistics)[0]}
    df = pd.DataFrame(*reversed(list(zip(*means.items()))))
    return df, evals                                           