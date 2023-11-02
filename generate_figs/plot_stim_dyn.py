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

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from scipy.stats import sem

from itertools import chain

from svm_cell_act import run_pca_all_model, run_SVM_all_model, load_sac_act
from landscape_analysis import (
    project_all_data,
    calculate_potential,
    aggregate_potential,
)

# plot settings
plt.rcParams["figure.figsize"] = [10, 4]
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
    "cutNonspec_model",
    "cutSpec_model",
]
pic_dir = os.path.join("new_dyn_analysis", "plots")
data_dir = os.path.join("new_dyn_analysis", "data")

total_rep = 50
replot_single_model = True


def main():
    all_act_mat_pca = run_pca_all_model(f_dirs, False)
    all_model_acc_li, all_coef_li, all_intercept_li = run_SVM_all_model(
        f_dirs, all_act_mat_pca, rerun_calc=False
    )
    # plot_SVM_example(all_act_mat_pca, all_coef_li, all_intercept_li, 3)
    all_proj_li = project_all_data(
        all_act_mat_pca,
        all_coef_li,
        all_intercept_li,
        all_model_acc_li,
        rerun_calc=False,
    )
    # find max and min in all_pos_li, define spatial bins
    min_pos = np.inf
    max_pos = -np.inf
    for i in range(len(all_proj_li)):
        for j in range(len(all_proj_li[i])):
            min_pos = min(min_pos, np.min(np.vstack(all_proj_li[i][j])))
            max_pos = max(max_pos, np.max(np.vstack(all_proj_li[i][j])))
    bins = np.linspace(min_pos, max_pos, 60)

    (
        all_potential_li,
        all_pos_li,
        all_mean_act_li,
        stim_label_map,
    ) = calc_all_stim_potential(all_proj_li, bins, rerun_calc=False)
    model_fig_dir = os.path.join(pic_dir, "stim_potential_stimOn200ms")
    if not os.path.exists(model_fig_dir):
        os.makedirs(model_fig_dir)
    all_stim_potential = []
    if replot_single_model:
        for rep in tqdm(range(total_rep)):
            rep_potential_li = [li[rep] for li in all_potential_li]
            rep_pos_li = [li[rep] for li in all_pos_li]
            rep_stim_potential = plot_model_stim_potential(
                rep,
                rep_potential_li,
                rep_pos_li,
                bins,
                (24, 34),
                stim_label_map,
                f_dirs,
                model_fig_dir,
            )
            all_stim_potential.append(rep_stim_potential)
        all_stim_potential = np.stack(all_stim_potential, axis=0)
        with open(os.path.join(data_dir, "all_stim_potential.pkl"), "wb") as f:
            dump(all_stim_potential, f)
    else:
        with open(os.path.join(data_dir, "all_stim_potential.pkl"), "rb") as f:
            all_stim_potential = load(f)
    plot_all_stim_potential(all_stim_potential, bins, stim_label_map, model_fig_dir)


def plot_all_stim_potential(all_stim_potential, bins, stim_label_map, model_fig_dir):
    color_arr = [
        "#3F6454",
        "#6D8B39",
        "#9CB027",
        "#F4DE0A",
        "#DF9F1E",
        "#D78724",
        "#C65734",
        "#944027",
    ]
    plt.figure(figsize=(10, 4))
    for i in range(len(all_stim_potential[0])):
        mean_potential = np.mean(all_stim_potential[:, i, :], axis=0)

        for i in range(len(stim_label_map)):
            mean_potential = np.mean(all_stim_potential[:, i], axis=0)
            sem_potential = sem(all_stim_potential[:, i], axis=0)
            plt.plot(
                bins[:-1],
                mean_potential,
                color=color_arr[i],
                label=list(stim_label_map.keys())[i + 1],
            )
            plt.fill_between(
                bins[:-1],
                mean_potential - sem_potential,
                mean_potential + sem_potential,
                color=color_arr[i],
                alpha=0.3,
            )
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("Potential")
    plt.tight_layout()
    plt.savefig(os.path.join(model_fig_dir, "all_stim_potential.png"), dpi=300)
    plt.close()


def plot_model_stim_potential(
    rep, potential_li, pos_li, bins, time_range, stim_label_map, f_dirs, model_fig_dir
):
    rep_stim_potential = []
    plt.figure(figsize=(10, 6))
    for i in range(len(potential_li)):
        plt.subplot(2, 2, i + 1)
        single_stim_potential = plot_single_stim_potential(
            potential_li[i], pos_li[i], bins, time_range, stim_label_map
        )
        rep_stim_potential.append(single_stim_potential)
        plt.title(f_dirs[i].split("_")[-2])
    plt.tight_layout()
    plt.savefig(os.path.join(model_fig_dir, "stim_potential_%d.png" % rep), dpi=300)
    plt.close()
    return np.stack(rep_stim_potential, axis=0)


def plot_single_stim_potential(potential, pos, bins, time_range, stim_label_map):
    color_arr = [
        "#3F6454",
        "#6D8B39",
        "#9CB027",
        "#F4DE0A",
        "#DF9F1E",
        "#D78724",
        "#C65734",
        "#944027",
    ]
    min_pos = np.inf
    max_pos = -np.inf
    all_mean_potential = []
    for i in range(len(potential)):
        binned_pos = np.digitize(pos[i], bins) - 1
        min_pos = min(min_pos, np.min(binned_pos))
        max_pos = max(max_pos, np.max(binned_pos))
        agg_potential = aggregate_potential(potential[i], binned_pos, bins)
        potential_seg = agg_potential[:, time_range[0] : time_range[1]]
        mean_potential = np.nanmean(potential_seg, axis=1)
        all_mean_potential.append(mean_potential)
        sem_potential = sem(potential_seg, axis=1, nan_policy="omit")
        plt.plot(
            bins,
            mean_potential,
            color=color_arr[i],
            label=list(stim_label_map.keys())[i],
        )
        plt.fill_between(
            bins,
            mean_potential - sem_potential,
            mean_potential + sem_potential,
            color=color_arr[i],
            alpha=0.3,
        )
    # plt.legend()
    plt.xlabel("Position")
    plt.ylabel("Potential")
    # plt.xticks(np.linspace(min_pos, max_pos, 5), labels=np.linspace(bins[min_pos], bins[max_pos], 5))
    return np.stack(all_mean_potential, axis=0)


def calc_all_stim_potential(all_proj_li, bins, rerun_calc=False):
    if rerun_calc:
        all_potential_li = []
        all_pos_li = []
        all_mean_act_li = []
        for m_idx, model_proj_li in enumerate(all_proj_li):
            model_potential_li = []
            model_pos_li = []
            model_mean_act_li = []
            for rep in tqdm(range(len(model_proj_li))):
                proj = model_proj_li[rep]
                stim_label, stim_label_map = get_stim_cond_idx(f_dirs[m_idx], rep, True)
                potential, positions, mean_act = calculate_potential(
                    proj, stim_label, bins
                )
                model_potential_li.append(potential)
                model_pos_li.append(positions)
                model_mean_act_li.append(mean_act)
            all_potential_li.append(model_potential_li)
            all_pos_li.append(model_pos_li)
            all_mean_act_li.append(model_mean_act_li)
        with open(os.path.join(data_dir, "stim_potential_li.pkl"), "wb") as f:
            dump(all_potential_li, f)
        with open(os.path.join(data_dir, "stim_pos_li.pkl"), "wb") as f:
            dump(all_pos_li, f)
        with open(os.path.join(data_dir, "stim_mean_act_li.pkl"), "wb") as f:
            dump(all_mean_act_li, f)
    else:
        with open(os.path.join(data_dir, "stim_potential_li.pkl"), "rb") as f:
            all_potential_li = load(f)
        with open(os.path.join(data_dir, "stim_pos_li.pkl"), "rb") as f:
            all_pos_li = load(f)
        with open(os.path.join(data_dir, "stim_mean_act_li.pkl"), "rb") as f:
            all_mean_act_li = load(f)
        with open(os.path.join(data_dir, "stim_label_map.pkl"), "rb") as f:
            stim_label_map = load(f)
    return all_potential_li, all_pos_li, all_mean_act_li, stim_label_map


def get_stim_cond_idx(
    f_dir,
    rep,
    reload,
    lr=0.02,
    plot_correct=True,
):
    if reload:
        n = SimpleNamespace(
            **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
        )
        coh_dict = find_coh_idx(n.stim_level)
        stim_label = np.zeros(len(n.stim_dir))
        stim_label_map = {}
        i = 1
        for coh in coh_dict.keys():
            for stim in np.unique(n.stim_dir):
                temp_key = "%s_%d" % (coh, stim)
                stim_label_map[temp_key] = i
                i += 1
        j = 1
        for coh in coh_dict.keys():
            for stim in np.unique(n.stim_dir):
                temp_idx = combine_idx(coh_dict[coh], n.stim_dir == stim)
                stim_label[temp_idx] = j
                j += 1
        if plot_correct:
            stim_label = stim_label[n.correct_idx]
        # save stim_label and stim_label_map
        with open(os.path.join(data_dir, "stim_label_rep%d.pkl" % rep), "wb") as f:
            dump(stim_label, f)
        if rep == 0:
            with open(os.path.join(data_dir, "stim_label_map.pkl"), "wb") as f:
                dump(stim_label_map, f)
        else:
            with open(os.path.join(data_dir, "stim_label_map.pkl"), "rb") as f:
                stim_label_map = load(f)
    else:
        with open(os.path.join(data_dir, "stim_label_rep%d.pkl" % rep), "rb") as f:
            stim_label = load(f)
        with open(os.path.join(data_dir, "stim_label_map.pkl"), "rb") as f:
            stim_label_map = load(f)
    return stim_label, stim_label_map


if __name__ == "__main__":
    main()
