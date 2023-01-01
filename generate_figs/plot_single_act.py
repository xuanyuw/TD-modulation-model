
import sys
import os
# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from types import SimpleNamespace
from utils import *

# plot settings
plt.rcParams['figure.figsize'] = [6, 4]
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['lines.linewidth'] = 2

f_dir = 'F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model'
rep = 0
lr = 0.02

stim_st_time = 45
target_st_time = 25

normalize = True

def main():
    n = SimpleNamespace(**load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep)))
    normalized_h = min_max_normalize(n.h)
    if normalize:
        h = normalized_h
    else:
        h = n.h
    
    # plot single neural activity
    motion_rng = np.concatenate((np.arange(0, 40), np.arange(80, 120), np.arange(160, 170), np.arange(180, 190)), axis=0)
    target_rng = np.concatenate((np.arange(40, 80), np.arange(120, 160), np.arange(170, 180), np.arange(190, 200)), axis=0)
    # for i in motion_rng:
    #     plot_avgAct_combined(h, n, i, True, mode='motion')
    for i in target_rng:
        if i in np.concatenate((np.arange(40, 80), np.arange(170, 180))):
            plot_avgAct_combined(h, n, i, True, mode='target', m1=True)
        else:
            plot_avgAct_combined(h, n, i, True, mode='target', m1=False)

###############################
# Plot single neuron activity #
###############################

def plot_avgAct_combined(h, n, cell_idx, save_plt, mode, m1=True):
    fig, ax = plt.subplots()
    coh_dict = find_coh_idx(n.stim_level)
    H_idx = coh_dict['H']
    M_idx = coh_dict['M']
    L_idx = coh_dict['L']
    Z_idx = coh_dict['Z']
    red_idx = n.stim_dir==315
    green_idx = n.stim_dir==135
    choice_color = get_choice_color(n.y, n.desired_out, n.stim_dir) # return choice color (green = 0, red = 1)
   
    # plot lines
    colors = {'H_red':'#FF0000', 'M_red': '#B30000', 'L_red': '#660000', 'H_green': '#00FF00', 'M_green': '#00B300', 'L_green':'#006600'}
    # zero coherence stimulus direction is based on the choice color
    if mode == "motion":
        if sum(Z_idx) != 0:
            ax.plot(np.mean(h[19:, combine_idx(Z_idx, choice_color[-1, :]==0), cell_idx], axis=1), linestyle ='--', color='#000000', label='135, Z')
            ax.plot(np.mean(h[19:, combine_idx(Z_idx, choice_color[-1, :]==1), cell_idx], axis=1), color='#000000', label='315, Z')
        ax.plot(np.mean(h[19:, combine_idx(M_idx, n.correct_idx, green_idx), cell_idx], axis=1), color=colors['M_green'], label='135, M')
        ax.plot(np.mean(h[19:, combine_idx(L_idx, n.correct_idx, green_idx), cell_idx], axis=1), color=colors['L_green'], label='135, L')
        ax.plot(np.mean(h[19:, combine_idx(H_idx, n.correct_idx, green_idx), cell_idx], axis=1), color=colors['H_green'], label='135, H')
        
        ax.plot(np.mean(h[19:, combine_idx(L_idx, n.correct_idx, red_idx), cell_idx], axis=1), color=colors['L_red'], label='315, L')
        ax.plot(np.mean(h[19:, combine_idx(M_idx, n.correct_idx, red_idx), cell_idx], axis=1), color=colors['M_red'], label='315, M')
        ax.plot(np.mean(h[19:, combine_idx(H_idx, n.correct_idx, red_idx), cell_idx], axis=1), color=colors['H_red'], label='315, H')
    elif mode == 'target':
        contra_idx, ipsi_idx = find_sac_idx(n.y, m1)
        if sum(Z_idx) != 0:
            ax.plot(np.mean(h[19:, combine_idx(Z_idx, contra_idx), cell_idx], axis=1), linestyle ='--', color='#000000', label='contra sac, Z')
            ax.plot(np.mean(h[19:, combine_idx(Z_idx, ipsi_idx), cell_idx], axis=1), color='#000000', label='ipsi sac, Z')
        ax.plot(np.mean(h[19:, combine_idx(L_idx, n.correct_idx, contra_idx), cell_idx], axis=1), color=colors['L_green'], label='contra sac, L')
        ax.plot(np.mean(h[19:, combine_idx(M_idx, n.correct_idx, contra_idx), cell_idx], axis=1), color=colors['M_green'], label='contra sac, M')
        ax.plot(np.mean(h[19:, combine_idx(H_idx, n.correct_idx, contra_idx), cell_idx], axis=1), color=colors['H_green'], label='contra sac, H')
        
        ax.plot(np.mean(h[19:, combine_idx(L_idx, n.correct_idx, ipsi_idx), cell_idx], axis=1), color=colors['L_red'], label='ipsi sac, L')
        ax.plot(np.mean(h[19:, combine_idx(M_idx, n.correct_idx, ipsi_idx), cell_idx], axis=1), color=colors['M_red'], label='ipsi sac, M')
        ax.plot(np.mean(h[19:, combine_idx(H_idx, n.correct_idx, ipsi_idx), cell_idx], axis=1), color=colors['H_red'], label='ipsi sac, H')
    
    
    ax.set_xlim(0, 50)
    xticks = np.array([0, 25, 50])
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks+20-stim_st_time)*20)

    ax.set_title('Cell %d, %s'% (cell_idx, mode))
    ax.set_ylabel("Average activity")
    ax.set_xlabel("Time")
    ax.axvline(x=target_st_time-20, color='k', alpha=0.8, linestyle='--', linewidth=1)
    ax.axvline(x=stim_st_time-20, color='k', alpha=0.8, linestyle='--', linewidth=1)
    ax.legend(loc='best', prop={'size': 10}, frameon=False)
    plt.tight_layout()

    if save_plt:
        pic_dir = os.path.join(f_dir, 'single_neuron_activity_rep%d_combined' %(rep))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir,'cell_%d.png'% cell_idx))
        plt.savefig(os.path.join(pic_dir,'cell_%d.pdf'% cell_idx))
        plt.close(fig)


main()