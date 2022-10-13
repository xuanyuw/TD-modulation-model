import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
from types import SimpleNamespace

f_dir = "crossOutput_noInterneuron_noMTConn_revSTP_model"
all_rep = range(20)
all_lr = [0.02]


def main(lr, rep):
    n = SimpleNamespace(**load_test_data(f_dir, lr, rep))
    # plot population neural activity
    normalized_h = min_max_normalize(n.h)
    m_idx = get_module_idx()
    motion_selective = pick_selective_neurons(normalized_h, n.stim_dir)
    saccade_selective = pick_selective_neurons(normalized_h, n.choice)
    title_arr = ["Motion", "Target"]

    m1_id = [[0, 4], [1, 5]]
    m2_id = [[2, 6], [3, 7]]


    # for i in range(len(title_arr)):
    i = 0
    plot_dir_selectivity(
        normalized_h,
        [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
        [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
        n,
        title_arr[i] + "_Motion_Selectivity_rep%d_selective"%rep,
        True,
        motion_selective,
    )
        # for i in range(len(title_arr)):
    plot_dir_selectivity(
        n.h,
        [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
        [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
        n,
        title_arr[i] + "_Motion_Selectivity_rep%d_selective_orig"%rep,
        True,
        motion_selective,
    )



    # for i in range(len(title_arr)):
    i = 1
    plot_sac_selectivity_pvnp(
        normalized_h,
        [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
        [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
        n,
        title_arr[i] + "_Saccade_Selectivity_rep%d_selective_pvnp"%rep,
        True,
        saccade_selective,
    )

    # for i in range(len(title_arr)):
    plot_sac_selectivity_pvnp(
        n.h,
        [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
        [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
        n,
        title_arr[i] + "_Saccade_Selectivity_rep%d_selective_pvnp_orig"%rep,
        True,
        saccade_selective,
    )

    # for i in range(len(title_arr)):
    i = 1
    plot_sac_selectivity_lvr(
        n.h,
        [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
        [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
        n,
        title_arr[i] + "_Saccade_Selectivity_rep%d_selective_lvr"%rep,
        True,
        saccade_selective,
    )

    # for i in range(len(title_arr)):
    plot_sac_selectivity_lvr(
        normalized_h,
        [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
        [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
        n,
        title_arr[i] + "_Saccade_Selectivity_rep%d_selective_lvr_norm"%rep,
        True,
        saccade_selective,
    )
    
    # for i in range(len(title_arr)):
    #     plot_dir_selectivity(
    #         normalized_h,
    #         [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
    #         [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
    #         n,
    #         title_arr[i] + "_Motion_Selectivity_rep%d"%rep,
    #         True,
    #     )


    # for i in range(len(title_arr)):
    #     plot_sac_selectivity_pvnp(
    #         normalized_h,
    #         [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
    #         [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
    #         n,
    #         title_arr[i] + "_Saccade_Selectivity_rep%d_pvnp"%rep,
    #         True,
    #     )

    # for i in range(len(title_arr)):
    #     plot_sac_selectivity_lvr(
    #         n.h,
    #         [m_idx[m1_id[i][0]], m_idx[m1_id[i][1]]],
    #         [m_idx[m2_id[i][0]], m_idx[m2_id[i][1]]],
    #         n,
    #         title_arr[i] + "_Saccade_Selectivity_rep%d_lvr"%rep,
    #         True,
    #     )




def plot_dir_selectivity(h, m1_idx, m2_idx, n, title, save_plt, selectivity=None):
    coh_dict = find_coh_idx(n.stim_level)
    # find the trial of preferred direction
    pref_red = find_pref_dir(n.stim_level, n.stim_dir, h)
    dir_red = n.stim_dir == 315
    pref_red_temp = np.tile(pref_red, (len(dir_red), 1))
    dir_red_temp = np.tile(np.reshape(dir_red, (-1, 1)), (1, len(pref_red)))
    pref_dir = pref_red_temp == dir_red_temp

    label_dict = {
        "dash": "nonPref",
        "solid": "Pref",
        "ax1_title": "Ipsi-lateral Saccade",
        "ax2_title": "Contra-lateral Saccade",
        "sup_title": title,
    }
    coh_levels = list(coh_dict.keys())
    line_dict = {}

    for coh in coh_levels:
        if coh == "Z":
            corr = None
        else:
            corr = n.correct_idx
        (
            line_dict["%s_solid_ax1" % coh],
            line_dict["%s_dash_ax1" % coh],
            line_dict["%s_solid_ax2" % coh],
            line_dict["%s_dash_ax2" % coh],
        ) = get_temp_h_avg(
            coh_dict[coh], h, n.y, pref_dir, m1_idx, m2_idx, "dir", corr, selectivity
        )
    fig = plot_coh_popu_act(line_dict, label_dict, coh_levels)
    if save_plt:
        folder_n = "popu_act"
        if selectivity is not None:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_rep%d_lr%f" % (folder_n, rep, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


def plot_sac_selectivity_pvnp(h, m1_idx, m2_idx, n, title, save_plt, selectivity=None):
    coh_dict = find_coh_idx(n.stim_level)
    # find the trial of preferred direction
    pref_ipsi, choice = find_pref_sac(n.y, h)
    pref_ipsi_temp = np.tile(pref_ipsi, (len(choice), 1))
    choice_temp = np.tile(np.reshape(choice, (-1, 1)), (1, len(pref_ipsi)))
    pref_sac = choice_temp == pref_ipsi_temp

    label_dict = {
        "dash": "nonPref",
        "solid": "Pref",
        "ax1_title": "Module 1",
        "ax2_title": "Module 2",
        "sup_title": title,
    }
    coh_levels = list(coh_dict.keys())
    line_dict = {}

    for coh in coh_levels:
        if coh == "Z":
            corr = None
        else:
            corr = n.correct_idx
        (
            line_dict["%s_solid_ax1" % coh],
            line_dict["%s_dash_ax1" % coh],
            line_dict["%s_solid_ax2" % coh],
            line_dict["%s_dash_ax2" % coh],
        ) = get_temp_h_avg(
            coh_dict[coh], h, n.y, pref_sac, m1_idx, m2_idx, "sac", corr, selectivity
        )
    fig = plot_coh_popu_act(line_dict, label_dict, coh_levels)
    if save_plt:
        folder_n = "popu_act"
        if selectivity is not None:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_rep%d_lr%f" % (folder_n, rep, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


def plot_sac_selectivity_lvr(h, m1_idx, m2_idx, n, title, save_plt, selectivity=None):
    coh_dict = find_coh_idx(n.stim_level)
    correct_idx = find_correct_idx(n.y, n.desired_out)

    label_dict = {
        "dash": "right",
        "solid": "left",
        "ax1_title": "Module 1",
        "ax2_title": "Module 2",
        "sup_title": title,
    }
    coh_levels = list(coh_dict.keys())
    line_dict = {}

    for coh in coh_levels:
        if coh == "Z":
            corr = None
        else:
            corr = correct_idx
        (
            line_dict["%s_solid_ax1" % coh],
            line_dict["%s_dash_ax1" % coh],
            line_dict["%s_solid_ax2" % coh],
            line_dict["%s_dash_ax2" % coh],
        ) = get_sac_avg_h(coh_dict[coh], h, n.choice, m1_idx, m2_idx, corr, selectivity)

    fig = plot_coh_popu_act(line_dict, label_dict, coh_levels)
    if save_plt:
        folder_n = "popu_act"
        if selectivity is not None:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_rep%d_lr%f" % (folder_n, rep, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


for rep in all_rep:
    for lr in all_lr:
# rep=4
# lr = 0.02
        main(lr, rep)
