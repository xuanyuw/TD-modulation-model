import sys
import os
# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from os.path import join
import numpy as np
import tables
from utils import pick_selective_neurons, min_max_normalize,recover_targ_loc, relu
from generate_figs.plot_weight_comp import locate_neurons
from calc_params import par
from scipy.stats import rankdata, ttest_1samp
from tqdm import tqdm

# plot settings

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['lines.linewidth'] = 2
# plt.rcParams['figure.figsize'] = [6, 4]


f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
plt_dir = os.path.join('generate_figs', 'Fig7', '7b_w_sel_corr')
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)


model_type = f_dir.split('_')[-2]
total_rep = 50
total_shuf = 100
lr = 2e-2
plot_sel = True
rerun_calculation = True
plot_trained = True
use_sel_rank =True
use_w_rank = True

title = "%s_weight_selectivity_correlation"%model_type

if plot_sel:
    title += '_selective'
else:
    title += '_allUnits'

if plot_trained:
    title += '_trained'
else:
    title += '_init'

if use_sel_rank:
    title += '_selRank'

if use_w_rank:
    title += '_wRank'

def calc_selectivity_val(normalized_h, labels, targ):
    # selectivity value > 0: prefer green, < 0: prefer red
    lbs = np.unique(labels)
    
    if targ:
        grp1_mean = np.mean(normalized_h[25:45, labels==lbs[0], :], axis = (0,1))
        grp2_mean = np.mean(normalized_h[25:45, labels==lbs[1], :], axis = (0,1))
        m1_rng = np.append(range(0, 80), range(160, 180))
        m2_rng = np.append(range(80, 160), range(180, 200))
        out = np.zeros(grp1_mean.shape)
        out[m1_rng] = (grp1_mean-grp2_mean)[m1_rng]
        out[m2_rng] = (grp2_mean-grp1_mean)[m2_rng]
        return out
    else:
        grp1_mean = np.mean(normalized_h[45:, labels==lbs[0], :], axis = (0,1))
        grp2_mean = np.mean(normalized_h[45:, labels==lbs[1], :], axis = (0,1))
        return grp1_mean-grp2_mean


def find_corr(from_arr, to_arr, mask, w, normalized_h, stim_labels, targ_labels, stim_rng):
    stim_sel_val = calc_selectivity_val(normalized_h, stim_labels, targ=False)
    targ_sel_val = calc_selectivity_val(normalized_h, targ_labels, targ=True)
    if use_sel_rank:
        stim_sel_val = rankdata(stim_sel_val, method='dense')
        targ_sel_val = rankdata(targ_sel_val, method='dense')
    non_zero_idx = np.argwhere(locate_neurons(from_arr, to_arr, mask))
    weights = []
    sel_diff = []
    for idx in non_zero_idx:
        if idx[0] in stim_rng:
            sel_diff.append(abs(stim_sel_val[idx[0]] - targ_sel_val[idx[1]])) 
        else:
            sel_diff.append(abs(targ_sel_val[idx[0]] - stim_sel_val[idx[1]])) 
        weights.append(w[idx[0], idx[1]])
    if use_w_rank:
        weights = rankdata(weights, method='dense')
        # reverse the rank from highest to lowest so lower weights will have a higher score.
        weights = len(weights)-weights

    return np.corrcoef(weights, sel_diff)[0, 1]


def load_data():
    if rerun_calculation:
    
        df = pd.DataFrame(columns=['corrcoef', 'conn_type', 'rep'])
        pbar = tqdm(total=total_rep)
        for rep in range(total_rep):
            # laod files
            if plot_trained:
                trained_w = np.load(join(f_dir, 'weight_%d_lr%f.pth'%(rep, lr)), allow_pickle=True)
                trained_w = trained_w.item()
            else:
                init_w = np.load(join(f_dir, 'init_weight_%d_lr%f.pth'%(rep, lr)), allow_pickle=True)
                init_w = init_w.item()
            test_output = tables.open_file(join(f_dir, 'test_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
            test_table = test_output.root
            normalized_h = min_max_normalize(test_table['h_iter0'][:])
            # find targ color encoding neurons 

            targ_loc = recover_targ_loc(test_table['target_iter0'][:], test_table['stim_dir_iter0'][:])[-1, :]
         

            targ_neurons = np.zeros((200,))
            targ_rng = np.concatenate((range(40, 80), range(170, 180), range(120, 160), range(190, 200)), axis=None)
            targ_neurons[targ_rng] = 1
            
            if plot_sel:
                targ_sel = pick_selective_neurons(normalized_h, targ_loc, window_st=25, window_ed=45, alpha=0.05)
                targ_neurons *= targ_sel
               
            stim_neurons = np.zeros((200,))
            stim_rng = np.concatenate((range(0, 40), range(160, 170), range(80, 120), range(180, 190)), axis=None)
            stim_neurons[stim_rng] = 1

            if plot_sel:
                stim_sel = pick_selective_neurons(normalized_h, test_table['stim_dir_iter0'][:], alpha=0.05)
                stim_neurons *= stim_sel
              
            if plot_trained:
                w_rnn =  par['EI_matrix'] @ relu(trained_w['w_rnn0'])
                rnn_mask = trained_w['rnn_mask_init']
            else:
                w_rnn = par['EI_matrix'] @ relu(init_w['w_rnn0'])
                rnn_mask = init_w['rnn_mask_init']

            # feedforward conn
            m2t_corr = find_corr(stim_neurons, targ_neurons, rnn_mask, w_rnn, normalized_h, test_table['stim_dir_iter0'][:], targ_loc, stim_rng)
            # feedback conn
            t2m_corr = find_corr(targ_neurons, stim_neurons, rnn_mask, w_rnn, normalized_h, test_table['stim_dir_iter0'][:], targ_loc, stim_rng)

            temp_df = pd.DataFrame({'corrcoef': [m2t_corr, t2m_corr], 'conn_type': ['ff', 'fb'],'rep': [rep]*2})
          
            df = pd.concat([df, temp_df])
            
            pbar.update(1)
        df.to_csv(join(f_dir, title+'_corr_data.csv'))
    else:
        df = pd.read_csv(join(f_dir, title+'_corr_data.csv'))
    return df

def plot_corr(df, title):
    df['conn_type'] = np.where(df['conn_type']=='ff', 'Feedforward', 'Feedback')

    colors = ['#FF0000','#0080FE']
    fig, ax = plt.subplots()
    sns.stripplot(x="conn_type", y="corrcoef", data=df, dodge=True, palette=colors, alpha=.8)
    # plot the mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'gray', 'ls': '--', 'lw': 2},
                width = 0.3,
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="conn_type",
                y="corrcoef",
                data=df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ax)
    plt.tight_layout()
    ax.set(xlabel="", ylabel="Correlation Coefficients")

    plt.savefig(join(plt_dir, '%s.png'%title), format='png')
    plt.savefig(join(plt_dir, '%s.pdf'%title), format='pdf')
    plt.savefig(join(plt_dir, '%s.eps'%title), format='eps')
    plt.close()

    # calculate mean and pVal
    ff_mean = df['corrcoef'][df['conn_type']=='Feedforward'].mean()
    fb_mean = df['corrcoef'][df['conn_type']=='Feedback'].mean()

    ff_pval = ttest_1samp(df['corrcoef'][df['conn_type']=='Feedforward'].to_numpy(), 0).pvalue
    fb_pval = ttest_1samp(df['corrcoef'][df['conn_type']=='Feedback'].to_numpy(), 0).pvalue

    with open(os.path.join(plt_dir, 'stat_test.txt'), 'w') as f:
        f.writelines('\n'.join(['Feedforward mean = %f, pval = %.4e'%(ff_mean, ff_pval), 'Feedback mean = %f, pval = %.4e'%(fb_mean, fb_pval)]))

def main():
    df = load_data()
    
    plot_corr(df, title)
   

        

if __name__ == '__main__':
    main()