import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import *
from types import SimpleNamespace
from pickle import load, dump
from tqdm import tqdm

# plot settings
plt.rcParams["figure.figsize"] = [10, 4]
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2

f_dir = "trained_withoutFB_model"
# model_type = f_dir
model_type = f_dir.split("_")[-2]
total_rep = 1
total_shuf = 0
all_lr = [2e-2]
plot_sel = True
rerun_calculation = True

if "shuf" in f_dir:
    plot_shuf = True
else:
    plot_shuf = False


def main(lr, total_rep):
    if rerun_calculation:
        dir_sel_norm, sac_sel_pvnp_norm, sac_sel_lvr_norm = load_all_activities(
            lr, total_rep, True, plot_sel, plot_shuf
        )
        # dir_sel_orig, sac_sel_pvnp_orig, sac_sel_lvr_orig  = load_all_activities(lr, total_rep, False, plot_sel)
    else:
        dir_sel_norm, sac_sel_pvnp_norm, sac_sel_lvr_norm = load(
            open(
                os.path.join(
                    f_dir, "all_selectivity_data_normalized_%dnet.pkl" % total_rep
                ),
                "rb",
            )
        )
        # dir_sel_norm, sac_sel_pvnp_norm, sac_sel_lvr_norm = load(open(os.path.join(f_dir, "all_selectivity_data_normalized.pkl"), 'rb'))

        if total_rep > 1 and not plot_shuf:
            for k in dir_sel_norm.keys():
                dir_sel_norm[k] = np.mean(dir_sel_norm[k], axis=0)
            for k in sac_sel_pvnp_norm.keys():
                sac_sel_pvnp_norm[k] = np.mean(sac_sel_pvnp_norm[k], axis=0)
            for k in sac_sel_lvr_norm.keys():
                sac_sel_lvr_norm[k] = np.mean(sac_sel_lvr_norm[k], axis=0)

        # dir_sel_orig, sac_sel_pvnp_orig, sac_sel_lvr_orig = load(open(os.path.join(f_dir, "all_selectivity_data_raw_%dnet.pkl"%total_rep), 'rb'))
        # for k in dir_sel_orig.keys():
        #     dir_sel_orig[k] = np.mean(dir_sel_orig[k], axis=0)
        # for k in sac_sel_pvnp_orig.keys():
        #     sac_sel_pvnp_orig[k] = np.mean(sac_sel_pvnp_orig[k], axis=0)
        # for k in sac_sel_lvr_orig.keys():
        #     sac_sel_lvr_orig[k] = np.mean(sac_sel_lvr_orig[k], axis=0)

    plot_dir_selectivity(
        dir_sel_norm, "%s_Motion_dir_sel_norm_avg" % model_type, True, plot_sel=plot_sel
    )
    # plot_dir_selectivity(dir_sel_orig, "%s_Motion_dir_sel_raw_avg"%model_type, True, plot_sel=plot_sel)

    plot_sac_selectivity_pvnp(
        sac_sel_pvnp_norm,
        "%s_Target_sac_sel_pvnp_norm_avg" % model_type,
        True,
        plot_sel=plot_sel,
    )
    # plot_sac_selectivity_pvnp(sac_sel_pvnp_orig, "%s_Target_sac_sel_pvnp_raw_avg"%model_type, True, plot_sel=plot_sel)

    plot_sac_selectivity_lvr(
        sac_sel_lvr_norm,
        "%s_Target_sac_sel_lvr_norm_avg" % model_type,
        True,
        plot_sel=plot_sel,
    )
    # plot_sac_selectivity_lvr(sac_sel_lvr_orig, "%s_Target_sac_sel_lvr_raw_avg"%model_type, True, plot_sel=plot_sel)


def load_single_activity(rep, lr, normalize, plot_sel, m_idx, m1_id, m2_id, shuf=None):
    if shuf is not None:
        n = SimpleNamespace(
            **load_test_data(
                f_dir, "test_output_lr%f_rep%d_shuf%d.h5" % (lr, rep, shuf)
            )
        )
    else:
        n = SimpleNamespace(
            **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
        )
    # plot population neural activity
    normalized_h = min_max_normalize(n.h)
    if normalize:
        h = normalized_h
    else:
        h = n.h

    if plot_sel:
        motion_selective = pick_selective_neurons(normalized_h, n.stim_dir)
        # saccade_selective = None
        saccade_selective = pick_selective_neurons(normalized_h, n.choice)
    else:
        motion_selective = None
        saccade_selective = None

    # plot motion selectiivty for motion module, and saccade selectivity for saccade selectivity for target module
    motion_dir_sel = calc_dir_sel(
        h,
        [m_idx[m1_id[0][0]], m_idx[m1_id[0][1]]],
        [m_idx[m2_id[0][0]], m_idx[m2_id[0][1]]],
        n,
        motion_selective,
    )
    sac_sel_pvnp = calc_sac_sel_pvnp(
        h,
        [m_idx[m1_id[1][0]], m_idx[m1_id[1][1]]],
        [m_idx[m2_id[1][0]], m_idx[m2_id[1][1]]],
        n,
        saccade_selective,
    )
    sac_sel_lvr = calc_sac_sel_lvr(
        h,
        [m_idx[m1_id[1][0]], m_idx[m1_id[1][1]]],
        [m_idx[m2_id[1][0]], m_idx[m2_id[1][1]]],
        n,
        saccade_selective,
    )

    return motion_dir_sel, sac_sel_pvnp, sac_sel_lvr


def load_all_activities(lr, total_rep, normalize, plot_sel, load_shuf=False):
    m_idx = get_module_idx()
    m1_id = [[0, 4], [1, 5]]
    m2_id = [[2, 6], [3, 7]]

    all_motion_dir_sel = {}
    all_sac_sel_pvnp = {}
    all_sac_sel_lvr = {}

    if load_shuf:
        pbar = tqdm(total=total_rep * total_shuf)
        for rep in range(total_rep):
            for shuf in range(total_shuf):
                motion_dir_sel, sac_sel_pvnp, sac_sel_lvr = load_single_activity(
                    rep, lr, normalize, plot_sel, m_idx, m1_id, m2_id, shuf
                )
                if rep == 0:
                    all_motion_dir_sel = motion_dir_sel
                    all_sac_sel_pvnp = sac_sel_pvnp
                    all_sac_sel_lvr = sac_sel_lvr
                else:
                    for k in motion_dir_sel.keys():
                        all_motion_dir_sel[k] = np.vstack(
                            [all_motion_dir_sel[k], motion_dir_sel[k]]
                        )
                    for k in sac_sel_pvnp.keys():
                        all_sac_sel_pvnp[k] = np.vstack(
                            [all_sac_sel_pvnp[k], sac_sel_pvnp[k]]
                        )
                    for k in sac_sel_lvr.keys():
                        all_sac_sel_lvr[k] = np.vstack(
                            [all_sac_sel_lvr[k], sac_sel_lvr[k]]
                        )
                pbar.update(1)
    else:
        pbar = tqdm(total=total_rep)
        for rep in range(total_rep):
            motion_dir_sel, sac_sel_pvnp, sac_sel_lvr = load_single_activity(
                rep, lr, normalize, plot_sel, m_idx, m1_id, m2_id
            )
            if rep == 0:
                all_motion_dir_sel = motion_dir_sel
                all_sac_sel_pvnp = sac_sel_pvnp
                all_sac_sel_lvr = sac_sel_lvr
            else:
                for k in motion_dir_sel.keys():
                    all_motion_dir_sel[k] = np.vstack(
                        [all_motion_dir_sel[k], motion_dir_sel[k]]
                    )
                for k in sac_sel_pvnp.keys():
                    all_sac_sel_pvnp[k] = np.vstack(
                        [all_sac_sel_pvnp[k], sac_sel_pvnp[k]]
                    )
                for k in sac_sel_lvr.keys():
                    all_sac_sel_lvr[k] = np.vstack([all_sac_sel_lvr[k], sac_sel_lvr[k]])
            pbar.update(1)

    if normalize:
        dump(
            [all_motion_dir_sel, all_sac_sel_pvnp, all_sac_sel_lvr],
            open(
                os.path.join(
                    f_dir, "all_selectivity_data_normalized_%dnet.pkl" % total_rep
                ),
                "wb",
            ),
        )
    else:
        dump(
            [all_motion_dir_sel, all_sac_sel_pvnp, all_sac_sel_lvr],
            open(
                os.path.join(f_dir, "all_selectivity_data_raw_%dnet.pkl" % total_rep),
                "wb",
            ),
        )
    if total_rep > 1:
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

    for k, v in line_dict.items():
        line_dict[k] = v[19:]

    fig = plot_coh_popu_act(line_dict, label_dict, ["Z", "L", "M", "H"])
    if save_plt:
        folder_n = "popu_act_%dnet" % total_rep
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg" % (folder_n))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.pdf" % title), format="pdf")
        plt.savefig(os.path.join(pic_dir, "%s.png" % title), format="png")
        plt.savefig(os.path.join(pic_dir, "%s.eps" % title), format="eps")
        plt.close(fig)


def plot_sac_selectivity_pvnp(line_dict, title, save_plt, plot_sel=False):
    label_dict = {
        "dash": "nonPref",
        "solid": "Pref",
        "ax1_title": "Module 1",
        "ax2_title": "Module 2",
        "sup_title": title,
    }
    for k, v in line_dict.items():
        line_dict[k] = v[19:]

    fig = plot_coh_popu_act(line_dict, label_dict, ["Z", "L", "M", "H"])
    if save_plt:
        folder_n = "popu_act_%dnet" % total_rep
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg" % (folder_n))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.pdf" % title), format="pdf")
        plt.savefig(os.path.join(pic_dir, "%s.png" % title), format="png")
        plt.savefig(os.path.join(pic_dir, "%s.eps" % title), format="eps")
        plt.close(fig)


def plot_sac_selectivity_lvr(line_dict, title, save_plt, plot_sel=False):
    label_dict = {
        "dash": "right",
        "solid": "left",
        "ax1_title": "Module 1",
        "ax2_title": "Module 2",
        "sup_title": title,
    }
    for k, v in line_dict.items():
        line_dict[k] = v[19:]
    fig = plot_coh_popu_act(line_dict, label_dict, ["Z", "L", "M", "H"])
    if save_plt:
        folder_n = "popu_act_%dnet" % total_rep
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg" % (folder_n))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.pdf" % title), format="pdf")
        plt.savefig(os.path.join(pic_dir, "%s.png" % title), format="png")
        plt.savefig(os.path.join(pic_dir, "%s.eps" % title), format="eps")
        plt.close(fig)


for lr in all_lr:
    main(lr, total_rep)
