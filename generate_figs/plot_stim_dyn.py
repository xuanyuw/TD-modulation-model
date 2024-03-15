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
from scipy.stats import sem
from statsmodels.formula.api import ols
import statsmodels.api as sm
import seaborn as sns
import pandas as pd

from svm_cell_act import run_pca_all_model, run_SVM_all_model
from landscape_analysis import project_all_data, calc_exp_speed, calc_cond_potential

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
    bins = np.linspace(min_pos, max_pos, 100)

    (
        all_potential_li,
        _,
        stim_label_map,
    ) = calc_all_stim_potential(all_proj_li, bins, rerun_calc=False)

    ANOVA2_pos_depth(all_potential_li, (40, 41))

    model_fig_dir = os.path.join(pic_dir, "stim_potential_stimOn200ms")
    if not os.path.exists(model_fig_dir):
        os.makedirs(model_fig_dir)
    plot_all_stim_potential(
        all_potential_li, bins, (40, 41), stim_label_map, model_fig_dir
    )


def ANOVA2_pos_depth(all_potential_li, time_range):
    all_potential_li = np.array(all_potential_li)[
        :, :, :, :, time_range[0] : time_range[1]
    ]
    potential_t = np.nanmean(all_potential_li, axis=4)
    df = pd.DataFrame(columns=["model", "coh", "min_potential", "max_pos"])
    for i in range(2):
        for rep in range(potential_t.shape[1]):
            for j in range(potential_t.shape[2]):
                max_pos = np.nanargmin(potential_t[i, rep, j, :])
                min_potential = np.nanmin(potential_t[i, rep, j, :])
                df = df.append(
                    {
                        "model": f_dirs[i].split("_")[-2],
                        "coh": j + 1,
                        "min_potential": min_potential,
                        "max_pos": max_pos,
                    },
                    ignore_index=True,
                )
    # 2way anova with coh and model
    model_potential = ols(
        "min_potential ~ C(coh) + C(model) + C(coh):C(model)", data=df
    ).fit()
    result1 = sm.stats.anova_lm(model_potential, type=2)
    print("\n")
    print(
        "Two-way ANOVA compare min potential: \n",
    )
    print(result1)
    # 2way anova with coh and model
    df["max_pos"] = df["max_pos"].astype(int)
    model_pos = ols("max_pos ~ C(coh) + C(model) + C(coh):C(model)", data=df).fit()
    result = sm.stats.anova_lm(model_pos, type=2)
    print("\n")
    print(
        "Two-way ANOVA compare max pos: \n",
    )
    print(result)

    df_diff = pd.DataFrame(columns=["model", "coh", "potential_diff", "pos_diff"])
    for i in range(2, 4):
        for rep in range(potential_t.shape[1]):
            for j in range(potential_t.shape[2]):
                max_pos_full = np.nanargmin(potential_t[0, rep, j, :])
                min_potential_full = np.nanmin(potential_t[0, rep, j, :])

                max_pos = np.nanargmin(potential_t[i, rep, j, :])
                min_potential = np.nanmin(potential_t[i, rep, j, :])

                pos_diff = max_pos_full - max_pos
                potential_diff = min_potential_full - min_potential

                df_diff = df_diff.append(
                    {
                        "model": f_dirs[i].split("_")[-2],
                        "coh": j + 1,
                        "potential_diff": potential_diff,
                        "pos_diff": pos_diff,
                    },
                    ignore_index=True,
                )

    model_potential_diff = ols(
        "potential_diff ~ C(coh) + C(model) + C(coh):C(model)", data=df_diff
    ).fit()
    result_potential_diff = sm.stats.anova_lm(model_potential_diff, type=2)
    print("\n")
    print(
        "Two-way ANOVA compare potential diff: \n",
    )
    print(result_potential_diff)

    df_diff["pos_diff"] = df_diff["pos_diff"].astype(int)
    model_pos_diff = ols(
        "pos_diff ~ C(coh) + C(model) + C(coh):C(model)", data=df_diff
    ).fit()
    result_pos_diff = sm.stats.anova_lm(model_pos_diff, type=2)
    print("\n")
    print(
        "Two-way ANOVA compare pos diff: \n",
    )
    print(result_pos_diff)


def flip_arr(arr, bins):
    if type(arr) == np.ma.core.MaskedArray:
        arr = arr.filled(np.nan)
    z_idx = np.digitize(0, bins)
    non_nan_idx = np.where(~np.isnan(arr))[0]
    non_nan_idx = non_nan_idx[non_nan_idx >= z_idx]
    arr_seg = np.flip(arr[non_nan_idx])
    flipped = np.ones_like(arr) * np.nan
    flipped[z_idx - len(non_nan_idx) + 1 : z_idx + 1] = arr_seg
    return flipped


def plot_stim_low_point(
    all_potential_li, bins, time_range, stim_label_map, model_fig_dir
):
    all_potential_li = np.array(all_potential_li)[
        :, :, :, :, time_range[0] : time_range[1]
    ]
    potential_t = np.nanmean(all_potential_li, axis=4)
    mean_potential = np.nanmean(potential_t, axis=1)
    sem_potential = sem(potential_t, axis=1, nan_policy="omit")
    min_idx = np.nanargmin(mean_potential, axis=2)
    # get the min potential and sem
    min_p = np.nanmin(mean_potential, axis=2)
    min_sem = np.empty_like(min_p)
    for i in range(min_p.shape[0]):
        for j in range(min_p.shape[1]):
            min_sem[i, j] = sem_potential[i, j, min_idx[i, j]]
    df = pd.DataFrame(columns=["model", "condition", "min_potential", "sem"])
    for i in range(min_p.shape[0]):
        for j in range(min_p.shape[1]):
            df = df.append(
                {
                    "model": f_dirs[i].split("_")[-2],
                    "condition": list(stim_label_map.keys())[j],
                    "min_potential": min_p[i, j],
                    "sem": min_sem[i, j],
                },
                ignore_index=True,
            )

    # plot bar with error bar sem
    plt.figure(figsize=(12, 6))
    g = sns.barplot(
        x="condition",
        y="min_potential",
        hue="model",
        data=df,
        palette=sns.color_palette("Set2"),
        capsize=0.1,
        order=[
            "H_c0",
            "M_c0",
            "L_c0",
            "Z_c0",
            "H_c1",
            "M_c1",
            "L_c1",
            "Z_c1",
        ],
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(model_fig_dir, "all_stim_potential_bar.png"), dpi=300)
    plt.close()


def plot_all_stim_potential(
    all_potential_li, bins, time_range, stim_label_map, model_fig_dir
):
    # TODO: fix positions
    color_arr = [
        "#3F6454",
        "#9CB027",
        "#6D8B39",
        "#F4DE0A",
    ]

    all_potential_li = np.array(all_potential_li)[
        :, :, :, :, time_range[0] : time_range[1]
    ]
    potential_t = np.nanmean(all_potential_li, axis=4)
    mean_potential = np.nanmean(potential_t, axis=1)
    sem_potential = sem(potential_t, axis=1, nan_policy="omit")
    z_idx = np.digitize(0, bins)

    for i in range(mean_potential.shape[0]):
        for j in range(mean_potential.shape[1]):
            temp = potential_t[i, :, j, :]
            # find positions with less than 4 valid values
            valid_sum = np.sum(~np.isnan(temp), axis=0)
            invalid_idx = np.where(valid_sum <= 3)[0]
            invalid_st_idx = min(invalid_idx[invalid_idx > z_idx])
            mean_potential[i, j, invalid_st_idx:] = np.nan
            sem_potential[i, j, invalid_st_idx:] = np.nan

    z_idx = np.digitize(0, bins)

    plt.figure(figsize=(12, 6))
    for i in range(potential_t.shape[0]):
        if i == 0:
            plt.subplot(2, 2, i + 1)
        else:
            plt.subplot(
                2, 2, i + 1, sharey=plt.subplot(2, 2, 1), sharex=plt.subplot(2, 2, 1)
            )
        for j in range(len(stim_label_map)):
            # if j % 2 == 0:
            #     mean_p = flip_arr(mean_potential[i, j, :], bins)
            #     sem_p = flip_arr(sem_potential[i, j, :], bins)
            # else:
            mean_p = mean_potential[i, j, :]
            sem_p = sem_potential[i, j, :]
            mean_p[:z_idx] = np.nan
            sem_p[:z_idx] = np.nan
            plt.plot(
                bins,
                mean_p,
                color=color_arr[j],
                label=list(stim_label_map.keys())[j],
            )
            # plt.ylim(np.nanmin(mean_potential) - 0.1, np.nanmax(mean_potential) + 0.1)

            plt.fill_between(
                bins,
                mean_p - sem_p,
                mean_p + sem_p,
                color=color_arr[j],
                alpha=0.3,
            )
        plt.xlabel("Position")
        plt.ylabel("Potential")
        plt.title(f_dirs[i].split("_")[-2])
    plt.legend()
    # plt.ylim(np.max(mean_potential) + 0.1, np.min(mean_potential) - 0.1)

    plt.tight_layout()
    # plt.savefig(os.path.join(model_fig_dir, "all_stim_potential.png"), dpi=300)
    plt.savefig(os.path.join(model_fig_dir, "all_stim_potential.pdf"), format="pdf")
    plt.close()


def calc_all_stim_potential(all_proj_li, bins, rerun_calc=False):
    if rerun_calc:
        all_potential_li = []
        all_mean_act_li = []
        for m_idx, model_proj_li in enumerate(all_proj_li):
            model_potential_li = []
            model_mean_act_li = []
            for rep in tqdm(range(len(model_proj_li))):
                proj = model_proj_li[rep]
                stim_label, stim_label_map = get_stim_cond_idx(f_dirs[m_idx], rep, True)
                potential, mean_act = calc_model_potential(proj, stim_label, bins)
                model_potential_li.append(potential)
                model_mean_act_li.append(mean_act)
            all_potential_li.append(model_potential_li)
            all_mean_act_li.append(model_mean_act_li)
        with open(os.path.join(data_dir, "stim_potential_li.pkl"), "wb") as f:
            dump(all_potential_li, f)
        with open(os.path.join(data_dir, "stim_mean_act_li.pkl"), "wb") as f:
            dump(all_mean_act_li, f)
    else:
        with open(os.path.join(data_dir, "stim_potential_li.pkl"), "rb") as f:
            all_potential_li = load(f)
        with open(os.path.join(data_dir, "stim_mean_act_li.pkl"), "rb") as f:
            all_mean_act_li = load(f)
        with open(os.path.join(data_dir, "stim_label_map.pkl"), "rb") as f:
            stim_label_map = load(f)
    return all_potential_li, all_mean_act_li, stim_label_map


def calc_model_potential(proj, label, bins):
    """
    Output:
        potential: 2d array, shape=(#bins, #time)
        mean_act: 2d array, shape=(#conditions, # time)
    """
    n_labels = len(np.unique(label))
    cond = np.arange(n_labels) + 1
    mean_act = []
    binned_proj = np.digitize(proj, bins) - 1
    potential = []

    all_exp_point_speed = np.nan * np.ones((len(bins), proj.shape[1] - 1))
    for c in cond:
        if sum(label == c) == 0:
            potential.append(np.nan * np.ones((len(bins), proj.shape[1] - 1)))
            mean_act.append(np.nan * np.ones((proj.shape[1])))
            continue

        binned_cond_proj = binned_proj[label == c]
        cond_mean = np.nanmean(binned_cond_proj, axis=0)

        x_diff = np.diff(binned_cond_proj, axis=1)
        exp_speed = calc_exp_speed(binned_cond_proj, x_diff, len(bins))
        cond_potential = calc_cond_potential(exp_speed, bins, c)
        potential.append(cond_potential)

        mean_act.append(cond_mean)
    mean_act = np.stack(mean_act, axis=0)
    return potential, mean_act


def get_stim_cond_idx(
    f_dir,
    rep,
    reload,
    lr=0.02,
    plot_correct=False,
):
    if reload:
        n = SimpleNamespace(
            **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
        )
        coh_dict = find_coh_idx(n.stim_level)
        stim_label = np.zeros(len(n.stim_level))
        stim_label_map = {}
        i = 1
        for coh in coh_dict.keys():
            # for c in np.unique(n.choice):
            # temp_key = "%s_c%d" % (coh, c)
            temp_key = coh
            stim_label_map[temp_key] = i
            i += 1
        j = 1
        for coh in coh_dict.keys():
            # for c in np.unique(n.choice):
            temp_idx = combine_idx(coh_dict[coh])
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
