import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
from types import SimpleNamespace

f_dir = "crossOutput_noInterneuron_noMTConn_2STF1STD_model"
total_rep = 20
all_lr = [0.02]



def main(lr, total_rep):
    dir_sel_norm, sac_sel_pvnp_norm, sac_sel_lvr_norm = load_all_activities(lr, total_rep, True, True)
    dir_sel_orig, sac_sel_pvnp_orig, sac_sel_lvr_orig  = load_all_activities(lr, total_rep, False, True)

    plot_dir_selectivity(dir_sel_norm, "Motion_direction_selectivity_normalized_average", True, plot_sel=True)
    plot_dir_selectivity(dir_sel_orig, "Motion_direction_selectivity_raw_average", True, plot_sel=True)
    
    plot_sac_selectivity_pvnp(sac_sel_pvnp_norm, "Target_saccade_selectivity_pvnp_normalized_average", True, plot_sel=True)
    plot_sac_selectivity_pvnp(sac_sel_pvnp_orig, "Target_saccade_selectivity_pvnp_raw_average", True, plot_sel=True)

    plot_sac_selectivity_lvr(sac_sel_lvr_norm, "Target_saccade_selectivity_lvr_normalized_average", True, plot_sel=True)
    plot_sac_selectivity_lvr(sac_sel_lvr_orig, "Target_saccade_selectivity_lvr_raw_average", True, plot_sel=True)

def load_all_activities(lr, total_rep, normalize, plot_sel):
    m_idx = get_module_idx()
    m1_id = [[0, 4], [1, 5]]
    m2_id = [[2, 6], [3, 7]]

    all_motion_dir_sel = {}
    all_sac_sel_pvnp = {}
    all_sac_sel_lvr = {}
    for rep in range(total_rep):
        n = SimpleNamespace(**load_test_data(f_dir, lr, rep))
        # plot population neural activity
        normalized_h = min_max_normalize(n.h)
        if normalize:
            h = normalized_h
        else:
            h = n.h
        
        if plot_sel:
            motion_selective = pick_selective_neurons(normalized_h, n.stim_dir)
            saccade_selective = pick_selective_neurons(normalized_h, n.choice)
        else:
            motion_selective = None
            saccade_selective = None
        
        # plot motion selectiivty for motion module, and saccade selectivity for saccade selectivity for target module 
        motion_dir_sel = calc_dir_sel(h, [m_idx[m1_id[0][0]], m_idx[m1_id[0][1]]], [m_idx[m2_id[0][0]], m_idx[m2_id[0][1]]], n, motion_selective)
        sac_sel_pvnp =  calc_sac_sel_pvnp(h, [m_idx[m1_id[1][0]], m_idx[m1_id[1][1]]], [m_idx[m2_id[1][0]], m_idx[m2_id[1][1]]], n, saccade_selective)
        sac_sel_lvr = calc_sac_sel_lvr(h, [m_idx[m1_id[1][0]], m_idx[m1_id[1][1]]], [m_idx[m2_id[1][0]], m_idx[m2_id[1][1]]], n, saccade_selective)

        if rep==0:
            all_motion_dir_sel = motion_dir_sel
            all_sac_sel_pvnp = sac_sel_pvnp
            all_sac_sel_lvr = sac_sel_lvr
        else:
            for k in motion_dir_sel.keys():
                all_motion_dir_sel[k] = np.vstack([all_motion_dir_sel[k],  motion_dir_sel[k]])
            for k in sac_sel_pvnp.keys():
                all_sac_sel_pvnp[k] = np.vstack([all_sac_sel_pvnp[k],  sac_sel_pvnp[k]])
            for k in sac_sel_lvr.keys():
                all_sac_sel_lvr[k] = np.vstack([all_sac_sel_lvr[k],  sac_sel_lvr[k]])

    for k in all_motion_dir_sel.keys():
        all_motion_dir_sel[k] = np.mean(all_motion_dir_sel[k], axis=0)
    for k in all_sac_sel_pvnp.keys():
        all_sac_sel_pvnp[k] = np.mean(all_sac_sel_pvnp[k], axis=0)
    for k in all_sac_sel_lvr.keys():
        all_sac_sel_lvr[k] = np.mean(all_sac_sel_lvr[k], axis=0)
    
    return all_motion_dir_sel, all_sac_sel_pvnp, all_sac_sel_lvr


def calc_dir_sel(h, m1_idx, m2_idx, n, selectivity=None):
    coh_dict = find_coh_idx(n.stim_level)
    # find the trial of preferred direction
    pref_red = find_pref_dir(n.stim_level, n.stim_dir, h)
    dir_red = n.stim_dir == 315
    pref_red_temp = np.tile(pref_red, (len(dir_red), 1))
    dir_red_temp = np.tile(np.reshape(dir_red, (-1, 1)), (1, len(pref_red)))
    pref_dir = pref_red_temp == dir_red_temp
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
    return line_dict

def calc_sac_sel_pvnp(h, m1_idx, m2_idx, n, selectivity=None):
    coh_dict = find_coh_idx(n.stim_level)
    # find the trial of preferred direction
    pref_ipsi, choice = find_pref_sac(n.y, h)
    pref_ipsi_temp = np.tile(pref_ipsi, (len(choice), 1))
    choice_temp = np.tile(np.reshape(choice, (-1, 1)), (1, len(pref_ipsi)))
    pref_sac = choice_temp == pref_ipsi_temp

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
    return line_dict

def calc_sac_sel_lvr(h, m1_idx, m2_idx, n, selectivity=None):
    coh_dict = find_coh_idx(n.stim_level)
    correct_idx = find_correct_idx(n.y, n.desired_out)
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
    return line_dict
    

def plot_dir_selectivity(line_dict, title, save_plt, plot_sel=False):

    label_dict = {
        "dash": "nonPref",
        "solid": "Pref",
        "ax1_title": "Ipsi-lateral Saccade",
        "ax2_title": "Contra-lateral Saccade",
        "sup_title": title,
    }

    fig = plot_coh_popu_act(line_dict, label_dict, ['Z', 'L', 'M', 'H'])
    if save_plt:
        folder_n = "popu_act"
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg_lr%f" % (folder_n, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


def plot_sac_selectivity_pvnp(line_dict, title, save_plt, plot_sel=False):

    label_dict = {
        "dash": "nonPref",
        "solid": "Pref",
        "ax1_title": "Module 1",
        "ax2_title": "Module 2",
        "sup_title": title,
    }

    fig = plot_coh_popu_act(line_dict, label_dict, ['Z', 'L', 'M', 'H'])
    if save_plt:
        folder_n = "popu_act"
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg_lr%f" % (folder_n, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


def plot_sac_selectivity_lvr(line_dict, title, save_plt, plot_sel=False):
    label_dict = {
        "dash": "right",
        "solid": "left",
        "ax1_title": "Module 1",
        "ax2_title": "Module 2",
        "sup_title": title,
    }

    fig = plot_coh_popu_act(line_dict, label_dict, ['Z', 'L', 'M', 'H'])
    if save_plt:
        folder_n = "popu_act"
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg_lr%f" % (folder_n, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)


for lr in all_lr:
    main(lr, total_rep)
