import sys
import os
# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pickle import load
import pandas as pd
from os.path import join, exists
from os import makedirs
from tables import open_file
from scipy.special import softmax
import numpy as np
from plot_utils import *

# plot settings
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['lines.linewidth'] = 2
# plt.rcParams['figure.figsize'] = [12, 5]


# sns.set_theme(context = 'paper', style='white', font = 'Arial')
# sns.set_context("paper", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":15, 'xtick.labelsize':12, 'ytick.labelsize':12})  

# mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['axes.spines.top'] = False
# mpl.rcParams.update({'font.size': 20})

# plt.rcParams['figure.figsize'] = [12, 8]
color_palette = {'H': '#FF0000', 'M': '#00FF00', 'L':'#0000FF', 'Z': 'k'}

save_plot = True
rerun_calc = False

total_rep = 50
root_dir = "F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
plot_dir = os.path.join('generate_figs', 'Fig5')
if not exists(plot_dir):
    makedirs(plot_dir)


if rerun_calc:
    # load all rt and acc into a dataframe
    all_acc_rt_df = pd.DataFrame(columns=['rep', 'coh', 'acc', 'rt'])
    for rep in range(total_rep):
        with open(join(root_dir, 'test_results_%d.pkl' %(rep)), 'rb') as f:
            data = load(f)
        rt_dict = calc_coh_rt(root_dir, rep, 0.8)

        temp_df = pd.DataFrame({
                    "rep": [rep] *4,
                    "coh":['H', 'M', 'L', 'Z'], 
                    "acc":[data['H_acc'][0], data['M_acc'][0], data['L_acc'][0], data['Z_acc'][0]],
                    "rt": [np.mean(rt_dict['H']), np.mean(rt_dict['M']), np.mean(rt_dict['L']), np.mean(rt_dict['Z'])]
                                })
        all_acc_rt_df = pd.concat([all_acc_rt_df, temp_df], ignore_index=True)
    all_acc_rt_df.to_csv(join(root_dir, 'all_acc_rt_df.csv'))
else:
    all_acc_rt_df = pd.read_csv(join(root_dir, 'all_acc_rt_df.csv'))

# fig 5c: violin plot of accuracy
fig, ax = plt.subplots()
sns.violinplot(x = 'coh', y = 'acc', data = pd.DataFrame(all_acc_rt_df.to_dict()), inner='points', palette = color_palette, ax=ax)
plt.setp(ax.collections, alpha = 0.5)
ax.set(xlabel='Coherence', ylabel="Accuracy")
ax.tick_params(bottom=True, left=True)
plt.tight_layout()
if save_plot:
    plt.savefig(join(plot_dir, "5c_acc.pdf"))

# fig 5d: violin plot of reaction time
fig, ax = plt.subplots()
sns.violinplot(x = 'coh', y = 'rt', data = pd.DataFrame(all_acc_rt_df.to_dict()), inner='points', palette = color_palette, ax=ax)
plt.setp(ax.collections, alpha = 0.5)
ax.set(xlabel='Coherence', ylabel="Reaction Time (ms)")
ax.tick_params(bottom=True, left=True)
if save_plot:
    plt.savefig(join(plot_dir, "5d_rt.pdf"))

