import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pickle import load
import pandas as pd
from os.path import join
import numpy as np
import tables
from utils import pick_selective_neurons, min_max_normalize,recover_targ_loc, relu
from plot_weight_comp import locate_neurons
from calc_params import par
from scipy.stats import rankdata

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
model_type = f_dir.split('_')[-2]
total_rep = 50
total_shuf = 100
lr = 2e-2
plot_sel = True
rerun_calculation = True
plot_trained = False
use_sel_rank =True
use_w_rank = False

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

    return np.corrcoef(weights, sel_diff)[0, 1]


def load_data():
    if rerun_calculation:
    
        df = pd.DataFrame(columns=['corrcoef', 'conn_type', 'rep'])
        
        for rep in range(total_rep):
            print('Loading rep {}'.format(rep))
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
        df.to_csv(join(f_dir, title+'_corr_data.csv'))
    else:
        df = pd.read_csv(join(f_dir, title+'_corr_data.csv'))
    return df

def plot_corr(df, title):
    p = sns.stripplot(x="conn_type", y="corrcoef", data=df, dodge=True, palette="dark", alpha=.6)
    # plot the mean line
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '-', 'lw': 2},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="conn_type",
                y="corrcoef",
                data=df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    plt.ylim(-0.25, 0.1)
    plt.title(title)
    fn_png = title + '.png'
    fn_pdf = title + '.pdf'

    plt.savefig(join(f_dir, fn_png))
    plt.savefig(join(f_dir, fn_pdf))
    plt.close()
    

def main():
    df = load_data()
    
    plot_corr(df, title)
   

        

if __name__ == '__main__':
    main()