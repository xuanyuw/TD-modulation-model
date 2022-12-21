import numpy as np
import tables
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
from utils import (
    find_coh_idx,
    find_correct_idx,
    find_pref_dir,
    find_sac_idx,
    get_module_idx,
    get_max_iter,
    correct_zero_coh,
    min_max_normalize,
)
from calc_params import par


f_dir = "F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
all_rep = range(1)
all_lr = [0.02]

stim_st_time = 45
target_st_time = 25


def main(lr, rep):

    test_output = tables.open_file(
        join(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep)), mode="r"
    )
    # train_table = train_output.root
    test_table = test_output.root
    max_iter = get_max_iter(test_table)
    h = test_table["h_iter%d" % max_iter][:]
    y = test_table["y_hist_iter%d" % max_iter][:]
    desired_out = test_table["target_iter%d" % max_iter][:]
    stim_level = test_table["stim_level_iter%d" % max_iter][:]
    stim_dir = test_table["stim_dir_iter%d" % max_iter][:]
    desired_out, stim_dir = correct_zero_coh(y, stim_level, stim_dir, desired_out)
    motion_rng = np.concatenate((np.arange(0, 40), np.arange(80, 120), np.arange(160, 170), np.arange(180, 190)))
    all_ROC = calc_all_ROC(h[:, :, motion_rng], y, desired_out, stim_level, stim_dir)
    

#     draw_ROC_plots(
#         all_ROC, m_idx[0], m_idx[2], "Motion_Excitatory_ROC", False,
#     )


# def draw_ROC_plots(all_ROC, m1_idx, m2_idx, title, save_plt):
#     m1_ROC
#     return


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


def calc_all_ROC(h, y, desired_out, stim_level, stim_dir):
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    contra_idx_m1, ipsi_idx_m1 = find_sac_idx(y, True)
    contra_idx_m2, ipsi_idx_m2 = find_sac_idx(y, False)
    # find the trial of preferred direction
    pref_red = find_pref_dir(stim_level, stim_dir, h, stim_st_time)
    dir_red = stim_dir == 315
    pref_red_temp = np.tile(pref_red, (len(dir_red), 1))
    dir_red_temp = np.tile(np.reshape(dir_red, (-1, 1)), (1, len(pref_red)))
    pref_dir = pref_red_temp == dir_red_temp
    correct_idx = find_correct_idx(y, desired_out)

    ipsi_ROC = np.zeros((h.shape[0], h.shape[2]))
    contra_ROC = np.zeros((h.shape[0], h.shape[2]))
    for i in range(h.shape[2]):
        temp_h = h[:, :, i]
        temp_pref = pref_dir[:, i]
        pref_h = temp_h[:, temp_pref]
        non_pref_h = temp_h[:, ~temp_pref]
        single_roc = rocN(pref_h, non_pref_h)
        ipsi_ROC[:, i] = single_roc
    return ipsi_ROC, contra_ROC


for lr in all_lr:
    for rep in all_rep:
        main(lr, rep)
