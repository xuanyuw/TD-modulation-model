import re
import numpy as np
import tables
import matplotlib.pyplot as plt
import os
from utils import *
from scipy.stats import f_oneway

f_dir = "gamma_inout_reinit_model"
all_rep = range(5)
all_lr = [0.02]

stim_st_time = 45
target_st_time = 25


def main(lr, rep):

    # train_output = tables.open_file(os.path.join(f_dir, 'train_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
    test_output = tables.open_file(
        os.path.join(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep)), mode="r"
    )
    # train_table = train_output.root
    test_table = test_output.root
    max_iter = get_max_iter(test_table)
    h = test_table["h_iter%d" % max_iter][:]
    y = test_table["y_hist_iter%d" % max_iter][:]
    choice = np.argmax(y, 2)[-1, :]
    desired_out = test_table["target_iter%d" % max_iter][:]
    stim_level = test_table["stim_level_iter%d" % max_iter][:]
    stim_dir = test_table["stim_dir_iter%d" % max_iter][:]
    desired_out, stim_dir = correct_zero_coh(y, stim_level, stim_dir, desired_out)
    correct_idx = find_correct_idx(y, desired_out)
    # plot population neural activity
    normalized_h = min_max_normalize(h)
    m_idx = get_module_idx()
    motion_selective = pick_selective_neurons(normalized_h, stim_dir)
    saccade_selective = pick_selective_neurons(normalized_h, choice)
    title_arr = [
        "Motion_Excitatory",
        "Target_Excitatory",
        "Motion_Inhibitory",
        "Target_Inhibitory",
    ]
    m1_id = [0, 1, 4, 5]
    m2_id = [2, 3, 6, 7]
    # for i in range(len(title_arr)):
    #     plot_selective_neural_act(
    #         h,
    #         saccade_selective,
    #         m_idx[m1_id[i]],
    #         m_idx[m2_id[i]],
    #         choice,
    #         correct_idx,
    #         stim_level,
    #         title_arr[i] + "_Saccade_Selectivity",
    #         True,
    #     )

    for i in range(len(title_arr)):
        plot_dir_selectivity(
            normalized_h,
            m_idx[m1_id[i]],
            m_idx[m2_id[i]],
            motion_selective,
            y,
            desired_out,
            stim_level,
            stim_dir,
            title_arr[i] + "_Motion_Selectivity",
            True,
        )


def pick_selective_neurons(h, labels, window_st=45, window_ed=70, alpha=0.01):
    lbs = np.unique(labels)
    grp1_idx = np.where(labels == lbs[0])[0]
    grp2_idx = np.where(labels == lbs[1])[0]
    if len(grp1_idx) != len(grp2_idx):
        idx_len = min(len(grp1_idx), len(grp2_idx))
        grp1_idx = grp1_idx[:idx_len]
        grp2_idx = grp2_idx[:idx_len]
    grp1 = np.mean(h[window_st:window_ed, grp1_idx, :], axis=0)
    grp2 = np.mean(h[window_st:window_ed, grp2_idx, :], axis=0)
    _, p_vals = f_oneway(grp1, grp2)
    return p_vals < alpha


def get_avg_h(coh_idx, h, choice, correct_idx=None):
    choice = choice.astype(bool)
    left_idx = combine_idx(~choice, coh_idx, correct_idx)
    right_idx = combine_idx(choice, coh_idx, correct_idx)

    h_left = np.mean(h[:, left_idx, :], axis=(1, 2))
    h_right = np.mean(h[:, right_idx, :], axis=(1, 2))

    return h_left, h_right


def create_grp_mask_sel(mask_shape, grp_idx, selectivity):
    mask = np.zeros(mask_shape).astype(bool)
    mask[:, cell_idx(grp_idx, selectivity)] = True
    return mask


def get_temp_h_sel(
    coh_idx, h, y, pref_dir, m1_idx, m2_idx, selectivity, correct_idx=None
):
    contra_idx_m1, ipsi_idx_m1 = find_sac_idx(y, True)
    contra_idx_m2, ipsi_idx_m2 = find_sac_idx(y, False)
    m1_mask = create_grp_mask_sel(pref_dir.shape, m1_idx, selectivity)
    m2_mask = create_grp_mask_sel(pref_dir.shape, m2_idx, selectivity)

    ipsi_pref_idx_m1 = (
        pref_dir
        * np.broadcast_to(
            combine_idx(ipsi_idx_m1, coh_idx, correct_idx)[:, None], pref_dir.shape
        )
        * m1_mask
    )
    ipsi_nonpref_idx_m1 = (
        (~pref_dir)
        * np.broadcast_to(
            combine_idx(ipsi_idx_m1, coh_idx, correct_idx)[:, None], pref_dir.shape
        )
        * m1_mask
    )
    ipsi_pref_idx_m2 = (
        pref_dir
        * np.broadcast_to(
            combine_idx(ipsi_idx_m2, coh_idx, correct_idx)[:, None], pref_dir.shape
        )
        * m2_mask
    )
    ipsi_nonpref_idx_m2 = (
        (~pref_dir)
        * np.broadcast_to(
            combine_idx(ipsi_idx_m2, coh_idx, correct_idx)[:, None], pref_dir.shape
        )
        * m2_mask
    )

    ipsi_h_pref = np.append(h[:, ipsi_pref_idx_m1], h[:, ipsi_pref_idx_m2], axis=1)
    ipsi_h_nonpref = np.append(
        h[:, ipsi_nonpref_idx_m1], h[:, ipsi_nonpref_idx_m2], axis=1
    )
    contra_pref_idx_m1 = (
        pref_dir
        * np.broadcast_to(
            combine_idx(contra_idx_m1, coh_idx, correct_idx)[:, None], pref_dir.shape,
        )
        * m1_mask
    )
    contra_nonpref_idx_m1 = (
        (~pref_dir)
        * np.broadcast_to(
            combine_idx(contra_idx_m1, coh_idx, correct_idx)[:, None], pref_dir.shape,
        )
        * m1_mask
    )
    contra_pref_idx_m2 = (
        pref_dir
        * np.broadcast_to(
            combine_idx(contra_idx_m2, coh_idx, correct_idx)[:, None], pref_dir.shape,
        )
        * m2_mask
    )
    contra_nonpref_idx_m2 = (
        (~pref_dir)
        * np.broadcast_to(
            combine_idx(contra_idx_m2, coh_idx, correct_idx)[:, None], pref_dir.shape,
        )
        * m2_mask
    )
    contra_h_pref = np.append(
        h[:, contra_pref_idx_m1], h[:, contra_pref_idx_m2], axis=1
    )
    contra_h_nonpref = np.append(
        h[:, contra_nonpref_idx_m1], h[:, contra_nonpref_idx_m2], axis=1
    )
    return ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref


def cell_idx(module_idx, selectivity):
    return np.intersect1d(
        np.arange(module_idx[0], module_idx[1]), np.where(selectivity)[0]
    )


def plot_selective_neural_act(
    h, selectivity, m1_idx, m2_idx, choice, correct_idx, stim_level, title, save_plt
):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    m1_h = h[:, :, cell_idx(m1_idx, selectivity)]
    m2_h = h[:, :, cell_idx(m2_idx, selectivity)]
    # zero coherence stimulus direction is based on the choice color
    if sum(Z_idx) != 0:
        h_left_m1, h_right_m1 = get_avg_h(Z_idx, m1_h, choice)
        h_left_m2, h_right_m2 = get_avg_h(Z_idx, m2_h, choice)
        ax1.plot(
            h_right_m1, linestyle="--", color="k", label="right, Z",
        )
        ax1.plot(h_left_m1, color="k", label="left, Z")
        ax2.plot(
            h_right_m2, linestyle="--", color="k", label="right, Z",
        )
        ax2.plot(h_left_m2, color="k", label="left, Z")

    h_left_m1, h_right_m1 = get_avg_h(L_idx, m1_h, choice, correct_idx)
    h_left_m2, h_right_m2 = get_avg_h(L_idx, m2_h, choice, correct_idx)
    ax1.plot(h_right_m1, linestyle="--", color="b", label="right, L")
    ax1.plot(h_left_m1, color="b", label="left, L")
    ax2.plot(h_right_m2, linestyle="--", color="b", label="right, L")
    ax2.plot(h_left_m2, color="b", label="left, L")

    h_left_m1, h_right_m1 = get_avg_h(M_idx, m1_h, choice, correct_idx)
    h_left_m2, h_right_m2 = get_avg_h(M_idx, m2_h, choice, correct_idx)
    ax1.plot(h_right_m1, linestyle="--", color="g", label="right, M")
    ax1.plot(h_left_m1, color="g", label="left, M")
    ax2.plot(h_right_m2, linestyle="--", color="g", label="right, M")
    ax2.plot(h_left_m2, color="g", label="left, M")

    h_left_m1, h_right_m1 = get_avg_h(H_idx, m1_h, choice, correct_idx)
    h_left_m2, h_right_m2 = get_avg_h(H_idx, m2_h, choice, correct_idx)
    ax1.plot(h_right_m1, linestyle="--", color="r", label="right, H")
    ax1.plot(h_left_m1, color="r", label="left, H")
    ax2.plot(h_right_m2, linestyle="--", color="r", label="right, H")
    ax2.plot(h_left_m2, color="r", label="left, H")

    ax1.set_title("Module 1")
    ax1.set_ylabel("Average activity")
    ax1.set_xlabel("Time")
    ax1.axvline(x=target_st_time, color="k")
    ax1.axvline(x=stim_st_time, color="k")

    ax2.set_title("Module 2")
    ax2.set_ylabel("Average activity")
    ax2.set_xlabel("Time")
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax2.axvline(x=target_st_time, color="k")
    ax2.axvline(x=stim_st_time, color="k")

    plt.suptitle(title)

    if save_plt:
        pic_dir = os.path.join(f_dir, "orig_selective_neuron_popu_act_rep%d" % (rep))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


def plot_dir_selectivity(
    h,
    m1_idx,
    m2_idx,
    selectivity,
    y,
    desired_out,
    stim_level,
    stim_dir,
    title,
    save_plt,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    # find the trial of preferred direction
    pref_red = find_pref_dir(stim_level, stim_dir, h, stim_st_time)
    dir_red = stim_dir == 315
    pref_red_temp = np.tile(pref_red, (len(dir_red), 1))
    dir_red_temp = np.tile(np.reshape(dir_red, (-1, 1)), (1, len(pref_red)))
    pref_dir = pref_red_temp == dir_red_temp
    correct_idx = find_correct_idx(y, desired_out)

    # zero coherence stimulus direction is based on the choice color
    if sum(Z_idx) != 0:
        ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref = get_temp_h_sel(
            Z_idx, h, y, pref_dir, m1_idx, m2_idx, selectivity
        )
        ax1.plot(
            np.mean(ipsi_h_nonpref, axis=1),
            linestyle="--",
            color="k",
            label="nonPref, Z",
        )
        ax1.plot(np.mean(ipsi_h_pref, axis=1), color="k", label="Pref, Z")
        ax2.plot(
            np.mean(contra_h_nonpref, axis=1),
            linestyle="--",
            color="k",
            label="nonPref, Z",
        )
        ax2.plot(np.mean(contra_h_pref, axis=1), color="k", label="Pref, Z")

    ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref = get_temp_h_sel(
        L_idx, h, y, pref_dir, m1_idx, m2_idx, selectivity, correct_idx
    )
    ax1.plot(
        np.mean(ipsi_h_nonpref, axis=1), linestyle="--", color="b", label="nonPref, L"
    )
    ax1.plot(np.mean(ipsi_h_pref, axis=1), color="b", label="Pref, L")
    ax2.plot(
        np.mean(contra_h_nonpref, axis=1), linestyle="--", color="b", label="nonPref, L"
    )
    ax2.plot(np.mean(contra_h_pref, axis=1), color="b", label="Pref, L")

    ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref = get_temp_h_sel(
        M_idx, h, y, pref_dir, m1_idx, m2_idx, selectivity, correct_idx
    )
    ax1.plot(
        np.mean(ipsi_h_nonpref, axis=1), linestyle="--", color="g", label="nonPref, M"
    )
    ax1.plot(np.mean(ipsi_h_pref, axis=1), color="g", label="Pref, M")
    ax2.plot(
        np.mean(contra_h_nonpref, axis=1), linestyle="--", color="g", label="nonPref, M"
    )
    ax2.plot(np.mean(contra_h_pref, axis=1), color="g", label="Pref, M")

    ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref = get_temp_h_sel(
        H_idx, h, y, pref_dir, m1_idx, m2_idx, selectivity, correct_idx
    )
    ax1.plot(
        np.mean(ipsi_h_nonpref, axis=1), linestyle="--", color="r", label="nonPref, H"
    )
    ax1.plot(np.mean(ipsi_h_pref, axis=1), color="r", label="Pref, H")
    ax2.plot(
        np.mean(contra_h_nonpref, axis=1), linestyle="--", color="r", label="nonPref, H"
    )
    ax2.plot(np.mean(contra_h_pref, axis=1), color="r", label="Pref, H")

    ax1.set_title("Ipsi-lateral Saccade")
    ax1.set_ylabel("Average activity")
    ax1.set_xlabel("Time")
    ax1.axvline(x=target_st_time, color="k")
    ax1.axvline(x=stim_st_time, color="k")

    ax2.set_title("Contra-lateral Saccade")
    ax2.set_ylabel("Average activity")
    ax2.set_xlabel("Time")
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax2.axvline(x=target_st_time, color="k")
    ax2.axvline(x=stim_st_time, color="k")

    plt.suptitle(title)

    if save_plt:
        pic_dir = os.path.join(f_dir, "temp_motion_selectivity_rep%d" % rep)
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        # plt.close(fig)


for rep in all_rep:
    for lr in all_lr:
        main(lr, rep)
