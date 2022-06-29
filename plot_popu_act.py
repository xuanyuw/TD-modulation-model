import numpy as np
import tables
import matplotlib.pyplot as plt
import os
from utils import *


f_dir = "fix_in_out_conn_plots"
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
    desired_out = test_table["target_iter%d" % max_iter][:]
    stim_level = test_table["stim_level_iter%d" % max_iter][:]
    stim_dir = test_table["stim_dir_iter%d" % max_iter][:]
    desired_out, stim_dir = correct_zero_coh(y, stim_level, stim_dir, desired_out)
    # plot population neural activity
    normalized_h = min_max_normalize(h)
    m_idx = get_module_idx()
    # plot_dir_selectivity(
    #     normalized_h,
    #     m_idx[0],
    #     m_idx[2],
    #     y,
    #     desired_out,
    #     stim_level,
    #     stim_dir,
    #     "Motion_Excitatory_Direction_Selectivity",
    #     True,
    # )
    # plot_dir_selectivity(
    #     normalized_h,
    #     m_idx[1],
    #     m_idx[3],
    #     y,
    #     desired_out,
    #     stim_level,
    #     stim_dir,
    #     "Target_Excitatory_Direction_Selectivity",
    #     True,
    # )
    # plot_dir_selectivity(
    #     normalized_h,
    #     m_idx[4],
    #     m_idx[6],
    #     y,
    #     desired_out,
    #     stim_level,
    #     stim_dir,
    #     "Motion_Inhibitory_Direction_Selectivity",
    #     True,
    # )
    # plot_dir_selectivity(
    #     normalized_h,
    #     m_idx[5],
    #     m_idx[7],
    #     y,
    #     desired_out,
    #     stim_level,
    #     stim_dir,
    #     "Target_Inhibitory_Direction_Selectivity",
    #     True,
    # )

    # plot_sac_selectivity(
    #     normalized_h,
    #     m_idx[0],
    #     m_idx[2],
    #     y,
    #     desired_out,
    #     stim_level,
    #     "Motion_Excitatory_Saccade_Selectivity",
    #     True,
    # )
    # plot_sac_selectivity(
    #     normalized_h,
    #     m_idx[1],
    #     m_idx[3],
    #     y,
    #     desired_out,
    #     stim_level,
    #     "Target_Excitatory_Saccade_Selectivity",
    #     True,
    # )
    # plot_sac_selectivity(
    #     normalized_h,
    #     m_idx[4],
    #     m_idx[6],
    #     y,
    #     desired_out,
    #     stim_level,
    #     "Motion_Inhibitory_Saccade_Selectivity",
    #     True,
    # )
    # plot_sac_selectivity(
    #     normalized_h,
    #     m_idx[5],
    #     m_idx[7],
    #     y,
    #     desired_out,
    #     stim_level,
    #     "Target_Inhibitory_Saccade_Selectivity",
    #     True,
    # )

    plot_sac_selectivity_temp(
        normalized_h,
        m_idx[0],
        m_idx[2],
        y,
        desired_out,
        stim_level,
        "Motion_Excitatory_Saccade_Selectivity_noPrefSacGrp",
        True,
    )
    plot_sac_selectivity_temp(
        normalized_h,
        m_idx[1],
        m_idx[3],
        y,
        desired_out,
        stim_level,
        "Target_Excitatory_Saccade_Selectivity_noPrefSacGrp",
        True,
    )
    plot_sac_selectivity_temp(
        normalized_h,
        m_idx[4],
        m_idx[6],
        y,
        desired_out,
        stim_level,
        "Motion_Inhibitory_Saccade_Selectivity_noPrefSacGrp",
        True,
    )
    plot_sac_selectivity_temp(
        normalized_h,
        m_idx[5],
        m_idx[7],
        y,
        desired_out,
        stim_level,
        "Target_Inhibitory_Saccade_Selectivity_noPrefSacGrp",
        True,
    )


def plot_dir_selectivity(
    h, m1_idx, m2_idx, y, desired_out, stim_level, stim_dir, title, save_plt
):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
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
        ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref = get_temp_h(
            Z_idx, h, y, pref_dir, m1_idx, m2_idx, "dir"
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

    ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref = get_temp_h(
        L_idx, h, y, pref_dir, m1_idx, m2_idx, "dir", correct_idx
    )
    ax1.plot(
        np.mean(ipsi_h_nonpref, axis=1), linestyle="--", color="b", label="nonPref, L"
    )
    ax1.plot(np.mean(ipsi_h_pref, axis=1), color="b", label="Pref, L")
    ax2.plot(
        np.mean(contra_h_nonpref, axis=1), linestyle="--", color="b", label="nonPref, L"
    )
    ax2.plot(np.mean(contra_h_pref, axis=1), color="b", label="Pref, L")

    ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref = get_temp_h(
        M_idx, h, y, pref_dir, m1_idx, m2_idx, "dir", correct_idx
    )
    ax1.plot(
        np.mean(ipsi_h_nonpref, axis=1), linestyle="--", color="g", label="nonPref, M"
    )
    ax1.plot(np.mean(ipsi_h_pref, axis=1), color="g", label="Pref, M")
    ax2.plot(
        np.mean(contra_h_nonpref, axis=1), linestyle="--", color="g", label="nonPref, M"
    )
    ax2.plot(np.mean(contra_h_pref, axis=1), color="g", label="Pref, M")

    ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref = get_temp_h(
        H_idx, h, y, pref_dir, m1_idx, m2_idx, "dir", correct_idx
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
    # ax2.legend()
    ax2.axvline(x=target_st_time, color="k")
    ax2.axvline(x=stim_st_time, color="k")

    plt.suptitle(title)

    if save_plt:
        pic_dir = os.path.join(
            f_dir, "new_population_neuron_activity_rep%d_lr%f" % (rep, lr)
        )
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        # plt.close(fig)


def plot_sac_selectivity(
    h, m1_idx, m2_idx, y, desired_out, stim_level, title, save_plt
):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    correct_idx = find_correct_idx(y, desired_out)
    # find the trial of preferred direction
    pref_ipsi, choice = find_pref_sac(y, h, stim_st_time)
    pref_ipsi_temp = np.tile(pref_ipsi, (len(choice), 1))
    choice_temp = np.tile(np.reshape(choice, (-1, 1)), (1, len(pref_ipsi)))
    pref_sac = choice_temp == pref_ipsi_temp

    # zero coherence stimulus direction is based on the choice color
    if sum(Z_idx) != 0:
        h_pref_m1, h_nonpref_m1, h_pref_m2, h_nonpref_m2 = get_temp_h(
            Z_idx, h, y, pref_sac, m1_idx, m2_idx, "sac"
        )
        ax1.plot(
            np.mean(h_nonpref_m1, axis=1), linestyle="--", color="k", label="nonPref, Z"
        )
        ax1.plot(np.mean(h_pref_m1, axis=1), color="k", label="pref, Z")
        ax2.plot(
            np.mean(h_nonpref_m2, axis=1), linestyle="--", color="k", label="nonPref, Z"
        )
        ax2.plot(np.mean(h_pref_m2, axis=1), color="k", label="pref, Z")

    h_pref_m1, h_nonpref_m1, h_pref_m2, h_nonpref_m2 = get_temp_h(
        L_idx, h, y, pref_sac, m1_idx, m2_idx, "sac", correct_idx
    )
    ax1.plot(
        np.mean(h_nonpref_m1, axis=1), linestyle="--", color="b", label="nonPref, L"
    )
    ax1.plot(np.mean(h_pref_m1, axis=1), color="b", label="pref, L")
    ax2.plot(
        np.mean(h_nonpref_m2, axis=1), linestyle="--", color="b", label="nonPref, L"
    )
    ax2.plot(np.mean(h_pref_m2, axis=1), color="b", label="pref, L")

    h_pref_m1, h_nonpref_m1, h_pref_m2, h_nonpref_m2 = get_temp_h(
        M_idx, h, y, pref_sac, m1_idx, m2_idx, "sac", correct_idx
    )
    ax1.plot(
        np.mean(h_nonpref_m1, axis=1), linestyle="--", color="g", label="nonPref, M"
    )
    ax1.plot(np.mean(h_pref_m1, axis=1), color="g", label="pref, M")
    ax2.plot(
        np.mean(h_nonpref_m2, axis=1), linestyle="--", color="g", label="nonPref, M"
    )
    ax2.plot(np.mean(h_pref_m2, axis=1), color="g", label="pref, M")

    h_pref_m1, h_nonpref_m1, h_pref_m2, h_nonpref_m2 = get_temp_h(
        H_idx, h, y, pref_sac, m1_idx, m2_idx, "sac", correct_idx
    )
    ax1.plot(
        np.mean(h_nonpref_m1, axis=1), linestyle="--", color="r", label="nonPref, H"
    )
    ax1.plot(np.mean(h_pref_m1, axis=1), color="r", label="pref, H")
    ax2.plot(
        np.mean(h_nonpref_m2, axis=1), linestyle="--", color="r", label="nonPref, H"
    )
    ax2.plot(np.mean(h_pref_m2, axis=1), color="r", label="pref, H")

    ax1.set_title("Module 1")
    ax1.set_ylabel("Average activity")
    ax1.set_xlabel("Time")
    ax1.axvline(x=target_st_time, color="k")
    ax1.axvline(x=stim_st_time, color="k")

    ax2.set_title("Module 2")
    ax2.set_ylabel("Average activity")
    ax2.set_xlabel("Time")
    # ax2.legend()
    ax2.axvline(x=target_st_time, color="k")
    ax2.axvline(x=stim_st_time, color="k")

    plt.suptitle(title)

    if save_plt:
        pic_dir = os.path.join(
            f_dir, "new_population_neuron_activity_rep%d_lr%f" % (rep, lr)
        )
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


def get_temp_h_new(coh_idx, h, choice, m1_idx, m2_idx, correct_idx=None):

    left_idx_m1 = combine_idx(~choice, coh_idx, correct_idx)
    right_idx_m1 = combine_idx(choice, coh_idx, correct_idx)
    left_idx_m2 = combine_idx(~choice, coh_idx, correct_idx)
    right_idx_m2 = combine_idx(choice, coh_idx, correct_idx)

    h_left_m1 = h[:, left_idx_m1, m1_idx[0] : m1_idx[1]]
    h_right_m1 = h[:, right_idx_m1, m1_idx[0] : m1_idx[1]]
    h_left_m2 = h[:, left_idx_m2, m2_idx[0] : m2_idx[1]]
    h_right_m2 = h[:, right_idx_m2, m2_idx[0] : m2_idx[1]]

    return h_left_m1, h_right_m1, h_left_m2, h_right_m2


def plot_sac_selectivity_temp(
    h, m1_idx, m2_idx, y, desired_out, stim_level, title, save_plt
):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    correct_idx = find_correct_idx(y, desired_out)
    choice = np.argmax(y, 2)[-1, :]

    # zero coherence stimulus direction is based on the choice color
    if sum(Z_idx) != 0:
        h_left_m1, h_right_m1, h_left_m2, h_right_m2 = get_temp_h_new(
            Z_idx, h, choice, m1_idx, m2_idx,
        )
        ax1.plot(
            np.mean(h_right_m1, axis=(1, 2)),
            linestyle="--",
            color="k",
            label="right, Z",
        )
        ax1.plot(np.mean(h_left_m1, axis=(1, 2)), color="k", label="left, Z")
        ax2.plot(
            np.mean(h_right_m2, axis=(1, 2)),
            linestyle="--",
            color="k",
            label="right, Z",
        )
        ax2.plot(np.mean(h_left_m2, axis=(1, 2)), color="k", label="left, Z")

    h_left_m1, h_right_m1, h_left_m2, h_right_m2 = get_temp_h_new(
        L_idx, h, choice, m1_idx, m2_idx, correct_idx
    )
    ax1.plot(
        np.mean(h_right_m1, axis=(1, 2)), linestyle="--", color="b", label="right, L"
    )
    ax1.plot(np.mean(h_left_m1, axis=(1, 2)), color="b", label="left, L")
    ax2.plot(
        np.mean(h_right_m2, axis=(1, 2)), linestyle="--", color="b", label="right, L"
    )
    ax2.plot(np.mean(h_left_m2, axis=(1, 2)), color="b", label="left, L")

    h_left_m1, h_right_m1, h_left_m2, h_right_m2 = get_temp_h_new(
        M_idx, h, choice, m1_idx, m2_idx, correct_idx
    )
    ax1.plot(
        np.mean(h_right_m1, axis=(1, 2)), linestyle="--", color="g", label="right, M"
    )
    ax1.plot(np.mean(h_left_m1, axis=(1, 2)), color="g", label="left, M")
    ax2.plot(
        np.mean(h_right_m2, axis=(1, 2)), linestyle="--", color="g", label="right, M"
    )
    ax2.plot(np.mean(h_left_m2, axis=(1, 2)), color="g", label="left, M")

    h_left_m1, h_right_m1, h_left_m2, h_right_m2 = get_temp_h_new(
        H_idx, h, choice, m1_idx, m2_idx, correct_idx
    )
    ax1.plot(
        np.mean(h_right_m1, axis=(1, 2)), linestyle="--", color="r", label="right, H"
    )
    ax1.plot(np.mean(h_left_m1, axis=(1, 2)), color="r", label="left, H")
    ax2.plot(
        np.mean(h_right_m2, axis=(1, 2)), linestyle="--", color="r", label="right, H"
    )
    ax2.plot(np.mean(h_left_m2, axis=(1, 2)), color="r", label="left, H")

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
        pic_dir = os.path.join(
            f_dir, "new_population_neuron_activity_rep%d_lr%f" % (rep, lr)
        )
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


for rep in all_rep:
    for lr in all_lr:
        main(lr, rep)
