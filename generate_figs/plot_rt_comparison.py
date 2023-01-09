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
import numpy as np
from plot_utils import calc_coh_rt
from statannotations.Annotator import Annotator
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import to_rgb
from tqdm import tqdm
from statsmodels.formula.api import ols
import statsmodels.api as sm

# plot settings
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.figsize'] = [12, 5]

f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_shufFeedback_model"
]
plt_dir = os.path.join('generate_figs', 'Fig7', '7d_ablation_rt_comp')
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)

total_rep = 50
total_shuf = 100
rerun_calculation = False

def main():
    all_rt_df = load_rt()
    df = all_rt_df[['rep', 'coh', 'rt', 'model']].groupby(['model', 'rep', 'coh']).mean().reset_index()

    fig, ax = plt.subplots()
    colors = ['#FF0000', '#00FF00', '#0000FF', '#424242']
    handles = []
    sns.violinplot(x = 'coh', y = 'rt', hue = 'model', data = df, inner='points', ax=ax,  palette=['.2', '.5'], hue_order=['Full model', 'No feedback', 'Shuffled feedback'], order=['H', 'M', 'L', 'Z'])

    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 3])
        if ind % 3 == 1:
            rgb = 0.4 + 0.6 * np.array(rgb)  # make whiter
        if ind % 3 == 2:
            rgb = 0.7 + 0.3 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))

    ax.legend(handles=[tuple(handles[::3]), tuple(handles[1::3]), tuple(handles[2::3])], labels=df["model"].astype('category').cat.categories.to_list(),
            handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower left', frameon=False)


    # add statistical test results
    pairs = [(('H', 'Full model'), ('H', 'No feedback')), (('H', 'Full model'), ('H', 'Shuffled feedback')),
                (('M', 'Full model'), ('M', 'No feedback')), (('M', 'Full model'), ('M', 'Shuffled feedback')), 
                (('L', 'Full model'), ('L', 'No feedback')), (('L', 'Full model'), ('L', 'Shuffled feedback')), 
                (('Z', 'Full model'), ('Z', 'No feedback')), (('Z', 'Full model'), ('Z', 'Shuffled feedback'))]

    f =  open(os.path.join(plt_dir, "stat_test.txt"), 'w') 
    sys.stdout = f

    annot = Annotator(ax, pairs, data=df, x='coh', y='rt', hue='model', order=['H', 'M', 'L', 'Z'])
    annot.configure(test='t-test_ind', text_format='star', loc='outside')
    annot.apply_and_annotate()


    ax.set(xlabel='Coherence', ylabel="Reaction Time (ms)")
    ax.tick_params(bottom=True, left=True)
    plt.tight_layout()

    plt.savefig(os.path.join(plt_dir, "ablation_rt_comp.pdf"), format='pdf')
    plt.savefig(os.path.join(plt_dir, "ablation_rt_comp.eps"), format='eps')
    plt.savefig(os.path.join(plt_dir, "ablation_rt_comp.png"), format='png')

    df_mean = df[['model', 'coh', 'rt']].groupby(['model', 'coh']).mean()
    df_mean.to_csv(os.path.join(plt_dir, "all_model_rt_mean.csv"))


    # Performing two-way ANOVA
    model = ols('rt ~ C(model) + C(coh) +\
    C(model):C(coh)',
                data=df[df['coh']!='Z']).fit()
    result = sm.stats.anova_lm(model, type=2)
    print('\n')
    print('Two-way ANOVA Result:')
    print(result)

    f.close()
    


def load_rt():
    if rerun_calculation:
        all_rt_df = pd.DataFrame(columns=['rep', 'shuf', 'coh', 'rt', 'model'])
        for f_dir in f_dirs:
            if 'shuf' in f_dir:
                pbar = tqdm(total = total_rep*total_shuf)
                for rep in range(total_rep):
                    for shuf in range(total_shuf):
                        rt_temp = convert_rt2df(calc_coh_rt(f_dir, rep, shuf), rep, shuf)
                        rt_temp['model'] = ['Shuffled feedback']*len(rt_temp.index)
                        all_rt_df = pd.concat([all_rt_df, rt_temp])
                        pbar.update(1)
            else:
                pbar = tqdm(total = total_rep*total_shuf)
                for rep in range(total_rep):
                    rt_temp = convert_rt2df(calc_coh_rt(f_dir, rep), rep)
                    if 'noFeedback' in f_dir:
                        rt_temp['model'] = ['No feedback']*len(rt_temp.index)
                    else:
                        rt_temp['model'] = ['Full model']*len(rt_temp.index)
                    all_rt_df = pd.concat([all_rt_df, rt_temp])
                    pbar.update(1)
        all_rt_df.to_csv(os.path.join(f_dirs[0], 'all_models_rt.csv'))
    else:
        all_rt_df = pd.read_csv(os.path.join(f_dirs[0], 'all_models_rt.csv'))
    return all_rt_df
                


def convert_rt2df(rt_dict, rep, shuf=np.nan):
    rt_df = pd.DataFrame(columns=['rep', 'shuf', 'coh', 'rt'])
    all_rt_arr = np.array([])
    coh_arr = []
    for coh in ['H', 'M', 'L', 'Z']:
        coh_arr.append([coh]*len(rt_dict[coh]))
        all_rt_arr = np.concatenate((all_rt_arr, rt_dict[coh]))
    rep_arr = [rep]*len(all_rt_arr)
    shuf_arr = [shuf]*len(all_rt_arr)
    coh_arr = sum(coh_arr, [])
    return pd.DataFrame({'rep': rep_arr, 'shuf': shuf_arr, 'coh': coh_arr, 'rt': all_rt_arr})
        
main()     