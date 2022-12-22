import sys
import os
# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import (
    find_coh_idx,
    find_sac_idx,
    combine_idx,
    load_test_data,
    min_max_normalize,
    get_pref_idx,
)
from types import SimpleNamespace
from pickle import dump, load
from tqdm import tqdm
from time import perf_counter
from scipy.stats import ttest_rel

# plot settings
plt.rcParams['figure.figsize'] = [6, 4]
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['lines.linewidth'] = 2


# f_dir = "F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
f_dir = "/Users/xuanyuwu/Documents/GitHub/TD-modulation-model/crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
all_rep = range(50)
lr = 0.02

stim_st_time = 45
target_st_time = 25
rerun_calc = True
normalize = True
sep_sac = False

def main():
    if rerun_calc:   
        if sep_sac:
            H_ipsi_dir_ROC = np.zeros((len(all_rep),70-19, 100) )
            H_contra_dir_ROC = np.zeros((len(all_rep),70-19, 100) )
            M_ipsi_dir_ROC = np.zeros((len(all_rep),70-19, 100) )
            M_contra_dir_ROC = np.zeros((len(all_rep),70-19, 100) )
            L_ipsi_dir_ROC = np.zeros((len(all_rep),70-19, 100) )
            L_contra_dir_ROC = np.zeros((len(all_rep),70-19, 100) )
            Z_ipsi_dir_ROC = np.zeros((len(all_rep),70-19, 100) )
            Z_contra_dir_ROC = np.zeros((len(all_rep),70-19, 100) )
            
            H_ipsi_sac_ROC = np.zeros((len(all_rep),70-19, 100) )
            H_contra_sac_ROC = np.zeros((len(all_rep),70-19, 100) )
            M_ipsi_sac_ROC = np.zeros((len(all_rep),70-19, 100) )
            M_contra_sac_ROC = np.zeros((len(all_rep),70-19, 100) )
            L_ipsi_sac_ROC = np.zeros((len(all_rep),70-19, 100) )
            L_contra_sac_ROC = np.zeros((len(all_rep),70-19, 100) )
            Z_ipsi_sac_ROC = np.zeros((len(all_rep),70-19, 100) )
            Z_contra_sac_ROC = np.zeros((len(all_rep),70-19, 100) )
        else:
            H_dir_ROC = np.zeros((len(all_rep),70-19, 100))
            M_dir_ROC = np.zeros((len(all_rep),70-19, 100))
            L_dir_ROC = np.zeros((len(all_rep),70-19, 100))
            Z_dir_ROC = np.zeros((len(all_rep),70-19, 100))
            H_sac_ROC = np.zeros((len(all_rep),70-19, 100))
            M_sac_ROC = np.zeros((len(all_rep),70-19, 100))
            L_sac_ROC = np.zeros((len(all_rep),70-19, 100))
            Z_sac_ROC = np.zeros((len(all_rep),70-19, 100))
        pbar = tqdm(total=len(all_rep)*2*4)
        for rep in all_rep:
            # print('Running ROC calculation for rep %d ... '%rep)
            n = SimpleNamespace(**load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep)))
            normalized_h = min_max_normalize(n.h)
            if normalize:
                h = normalized_h
            else:
                h = n.h
            motion_rng = np.concatenate((np.arange(0, 40), np.arange(80, 120), np.arange(160, 170), np.arange(180, 190)))
            m1_rng = np.concatenate((np.arange(0, 40), np.arange(160, 170)))
            coh_dict = find_coh_idx(n.stim_level)
            H_idx = coh_dict['H']
            M_idx = coh_dict['M']
            L_idx = coh_dict['L']
            Z_idx = coh_dict['Z']
            pref_dir, pref_sac = get_pref_idx(n, h)
            if sep_sac:
                H_ipsi_dir_ROC[rep, :, :], H_contra_dir_ROC[rep, :, :] = calc_sac_sep_ROC(h[19:, :, motion_rng], n, m1_rng, H_idx, pref_dir)
                pbar.update(1)
                M_ipsi_dir_ROC[rep, :, :], M_contra_dir_ROC[rep, :, :] = calc_sac_sep_ROC(h[19:, :, motion_rng], n, m1_rng, M_idx, pref_dir)
                pbar.update(1)
                L_ipsi_dir_ROC[rep, :, :], L_contra_dir_ROC[rep, :, :] = calc_sac_sep_ROC(h[19:, :, motion_rng], n, m1_rng, L_idx, pref_dir)
                pbar.update(1)
                Z_ipsi_dir_ROC[rep, :, :], Z_contra_dir_ROC[rep, :, :] = calc_sac_sep_ROC(h[19:, :, motion_rng], n, m1_rng, Z_idx, pref_dir)
                pbar.update(1)
                
                H_ipsi_sac_ROC[rep, :, :], H_contra_sac_ROC[rep, :, :] = calc_sac_sep_ROC(h[19:, :, motion_rng], n, m1_rng, H_idx, pref_sac)
                pbar.update(1)
                M_ipsi_sac_ROC[rep, :, :], M_contra_sac_ROC[rep, :, :] = calc_sac_sep_ROC(h[19:, :, motion_rng], n, m1_rng, M_idx, pref_sac)
                pbar.update(1)
                L_ipsi_sac_ROC[rep, :, :], L_contra_sac_ROC[rep, :, :] = calc_sac_sep_ROC(h[19:, :, motion_rng], n, m1_rng, L_idx, pref_sac)
                pbar.update(1)
                Z_ipsi_sac_ROC[rep, :, :], Z_contra_sac_ROC[rep, :, :] = calc_sac_sep_ROC(h[19:, :, motion_rng], n, m1_rng, Z_idx, pref_sac)
                pbar.update(1)
            else:
                
                H_dir_ROC[rep, :, :] = calc_ROC(h[19:, :, motion_rng], n, H_idx, pref_dir)
                pbar.update(1)
                M_dir_ROC[rep, :, :] = calc_ROC(h[19:, :, motion_rng], n, M_idx, pref_dir)
                pbar.update(1)
                L_dir_ROC[rep, :, :] = calc_ROC(h[19:, :, motion_rng], n, L_idx, pref_dir)
                pbar.update(1)
                Z_dir_ROC[rep, :, :] = calc_ROC(h[19:, :, motion_rng], n, Z_idx, pref_dir)
                pbar.update(1)
                
                H_sac_ROC[rep, :, :] = calc_ROC(h[19:, :, motion_rng], n, H_idx, pref_sac)
                pbar.update(1)
                M_sac_ROC[rep, :, :] = calc_ROC(h[19:, :, motion_rng], n, M_idx, pref_sac)
                pbar.update(1)
                L_sac_ROC[rep, :, :] = calc_ROC(h[19:, :, motion_rng], n, L_idx, pref_sac)
                pbar.update(1)
                Z_sac_ROC[rep, :, :] = calc_ROC(h[19:, :, motion_rng], n, Z_idx, pref_sac)
                pbar.update(1)
        
        pbar.close()
        if sep_sac:
            with open(os.path.join(f_dir, 'sep_sac_ROC.pkl'), 'wb') as f:
                dump([H_ipsi_dir_ROC, H_contra_dir_ROC, M_ipsi_dir_ROC, M_contra_dir_ROC, L_ipsi_dir_ROC, L_contra_dir_ROC, Z_ipsi_dir_ROC, Z_contra_dir_ROC, 
                H_ipsi_sac_ROC, H_contra_sac_ROC, M_ipsi_sac_ROC, M_contra_sac_ROC, L_ipsi_sac_ROC, L_contra_sac_ROC, Z_ipsi_sac_ROC, Z_contra_sac_ROC], f)
        else:
            with open(os.path.join(f_dir, 'all_ROC.pkl'), 'wb') as f:
                dump([H_dir_ROC, M_dir_ROC, L_dir_ROC, Z_dir_ROC, H_sac_ROC, M_sac_ROC, L_sac_ROC, Z_sac_ROC], f)
    else:
        if sep_sac:
            with open(os.path.join(f_dir, 'sep_sac_ROC.pkl'), 'rb') as f:
                H_ipsi_dir_ROC, H_contra_dir_ROC, M_ipsi_dir_ROC, M_contra_dir_ROC, L_ipsi_dir_ROC, L_contra_dir_ROC, Z_ipsi_dir_ROC, Z_contra_dir_ROC, 
                H_ipsi_sac_ROC, H_contra_sac_ROC, M_ipsi_sac_ROC, M_contra_sac_ROC, L_ipsi_sac_ROC, L_contra_sac_ROC, Z_ipsi_sac_ROC, Z_contra_sac_ROC = load(f)
        else:
            with open(os.path.join(f_dir, 'all_ROC.pkl'), 'rb') as f:
                [H_dir_ROC, M_dir_ROC, L_dir_ROC, Z_dir_ROC, H_sac_ROC, M_sac_ROC, L_sac_ROC, Z_sac_ROC] = load(f)
    if not sep_sac:
        plot_all_avg_ROC(H_dir_ROC, M_dir_ROC, L_dir_ROC, Z_dir_ROC, 'dir')
        plot_all_avg_ROC(H_sac_ROC, M_sac_ROC, L_sac_ROC, Z_sac_ROC, 'sac')


def rocN(x, y, N=100):
    
    x = x.flatten("F")
    y = y.flatten("F")
    zlo = min(min(x), min(y))
    zhi = max(max(x), max(y))
    z = np.linspace(zlo, zhi, N)
    fa = np.zeros((N, ))
    hit = np.zeros((N, ))
    for i in range(N):
        fa[N - (i+1)] = sum(y > z[i])
        hit[N - (i+1)] = sum(x > z[i])

    fa = fa / y.shape[0]
    hit = hit / x.shape[0]
    a = np.trapz(y=hit, x=fa)

    return a


def calc_ROC(h, n, coh_idx, pref_idx):
    # tic = perf_counter()
    all_ROC = np.zeros((h.shape[0], h.shape[2]))
    for i in range(h.shape[2]):
        pre_idx = combine_idx(pref_idx[:, i], n.correct_idx,coh_idx)
        non_idx = combine_idx(~pref_idx[:, i], n.correct_idx,coh_idx)
        for j in range(h.shape[0]):
            h_pre = h[j, pre_idx, i]
            h_non = h[j, non_idx, i]
            all_ROC[j, i] = rocN(h_pre, h_non)
    # toc = perf_counter()
    # print(f"ROC ran in {toc - tic:0.4f} seconds")
    return all_ROC
   

def calc_sac_sep_ROC(h, n, m1_rng, coh_idx, pref_idx):
    
    contra_idx_m1, ipsi_idx_m1 = find_sac_idx(n.y, True)
    contra_idx_m2, ipsi_idx_m2 = find_sac_idx(n.y, False)

    ipsi_ROC = np.zeros((h.shape[0], h.shape[2]))
    contra_ROC = np.zeros((h.shape[0], h.shape[2]))
    for i in range(h.shape[2]):
        if i in m1_rng:
            contra_idx = contra_idx_m1
            ipsi_idx = ipsi_idx_m1
        else:
            contra_idx = contra_idx_m2
            ipsi_idx = ipsi_idx_m2
        ipsi_pref_idx = combine_idx(ipsi_idx, pref_idx[:, i], n.correct_idx,coh_idx)
        contra_pref_idx = combine_idx(contra_idx, pref_idx[:, i], n.correct_idx,coh_idx)
        ipsi_non_idx = combine_idx(ipsi_idx, ~pref_idx[:, i], n.correct_idx,coh_idx)
        contra_non_idx = combine_idx(contra_idx, ~pref_idx[:, i], n.correct_idx,coh_idx)

        h_ipsi_pref = h[:, ipsi_pref_idx, i]
        h_contra_pref = h[:, contra_pref_idx, i]
        h_ipsi_non = h[:, ipsi_non_idx, i]
        h_contra_non = h[:, contra_non_idx, i]
        ipsi_ROC[:, i] = rocN(h_ipsi_pref, h_ipsi_non)
        contra_ROC[:, i] = rocN(h_contra_pref, h_contra_non)
    return ipsi_ROC, contra_ROC


def plot_all_avg_ROC(H_ROC, M_ROC, L_ROC, Z_ROC, mode, cell_idx=None, save_plt=True):
    if cell_idx is None:
        H_line = np.mean(H_ROC, axis=(0, 2))
        M_line = np.mean(M_ROC, axis=(0, 2))
        L_line = np.mean(L_ROC, axis=(0, 2))
        Z_line = np.mean(Z_ROC, axis=(0, 2))

    else:
        H_line = H_ROC[0,cell_idx,:]
        M_line = M_ROC[0,cell_idx,:]
        L_line = L_ROC[0,cell_idx,:]
        Z_line = Z_ROC[0,cell_idx,:]



    fig, ax = plt.subplots()
    ax.plot(H_line, label='H', color='r')
    ax.plot(M_line, label='M', color='g')
    ax.plot(L_line, label='L', color='b')
    ax.plot(Z_line, label='Z', color='k')
    ax.legend(loc='best', prop={'size': 10}, frameon=False)

    ax.set_xlim(0, 50)
    xticks = np.array([0, 25, 50])
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks+20-stim_st_time)*20)
    ax.set_ylabel("Average AUC")
    ax.set_xlabel("Time")
    ax.axvline(x=target_st_time-20, color='k', alpha=0.8, linestyle='--', linewidth=1)
    ax.axvline(x=stim_st_time-20, color='k', alpha=0.8, linestyle='--', linewidth=1)
    plt.tight_layout()

    if save_plt:
        pic_dir = os.path.join(f_dir, 'ROC_plots')
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir,'all_ROC_%s.png'%mode))
        plt.savefig(os.path.join(pic_dir,'all_ROC_%s.pdf'%mode))
        plt.close(fig)



main()
