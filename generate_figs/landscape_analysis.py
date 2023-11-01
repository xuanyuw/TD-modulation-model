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
replot_single_model = False


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
    # plot_proj_example(all_proj_li, 3)
    all_potential_li, all_pos_li, all_mean_act_li = calc_all_potential(
        all_proj_li, rerun_calc=False
    )
    all_potential_agg = []
    model_fig_dir = os.path.join(pic_dir, "temp_landscape")
    if not os.path.exists(model_fig_dir):
        os.makedirs(model_fig_dir)
    for rep in tqdm(range(total_rep)):
        if replot_single_model:
            rep_potential_li = [li[rep] for li in all_potential_li]
            rep_pos_li = [li[rep] for li in all_pos_li]
            rep_mean_act_li = [li[rep] for li in all_mean_act_li]
            rep_potential_agg = plot_model_potential_heatmap(
                rep,
                rep_potential_li,
                rep_pos_li,
                rep_mean_act_li,
                model_fig_dir,
                data_dir,
                f_dirs,
            )
        else:
            rep_potential_agg = []
            for i in range(len(f_dirs)):
                with open(
                    os.path.join(data_dir, "rep_potential_agg_%s.pkl" % str(rep)), "rb"
                ) as f:
                    rep_potential_agg.append(load(f)[i])
            rep_potential_agg = np.stack(rep_potential_agg, axis=0)
        all_potential_agg.append(rep_potential_agg)
    all_potential_agg = np.stack(all_potential_agg, axis=0)
    avg_potential_agg = np.nanmean(all_potential_agg, axis=0)
    avg_mean_act = np.mean(np.array(all_mean_act_li), axis=1)

    # find max and min in all_pos_li
    min_pos = np.inf
    max_pos = -np.inf
    for i in range(len(all_pos_li)):
        for j in range(len(all_pos_li[i])):
            min_pos = min(min_pos, np.min(np.vstack(all_pos_li[i][j])))
            max_pos = max(max_pos, np.max(np.vstack(all_pos_li[i][j])))
    bins = np.linspace(min_pos, max_pos, 60)

    plot_mean_potential_heatmap(avg_potential_agg, avg_mean_act, bins, model_fig_dir)

    # for i in range(len(f_dirs)):
    #     model_fig_dir = os.path.join(
    #         pic_dir, "temp_landscape_%s" % (f_dirs[i].split("_")[-2])
    #     )
    #     if not os.path.exists(model_fig_dir):
    #         os.makedirs(model_fig_dir)
    #     plot_model_potential_heatmap(
    #         all_potential_li[i], all_pos_li[i], all_mean_act_li[i], model_fig_dir
    #     )


def plot_proj_example(all_proj_li, rep):
    _, _, label = load_sac_act(f_dirs[0], rep, reload=False)
    example = all_proj_li[0][rep][:, -1]
    plt.scatter(example[label == 0], np.ones(sum(label == 0)), c="g")
    plt.scatter(example[label == 1], np.ones(sum(label == 1)) * 2, c="r")
    plt.show()


def plot_SVM_example(all_act_mat_pca, all_coef_li, all_intercept_li, rep):
    _, _, label = load_sac_act(f_dirs[0], rep, reload=False)
    plot_SVM(
        all_act_mat_pca[0][rep][:, :, -1],
        label,
        all_coef_li[0][-1][0, :, rep],
        all_intercept_li[0][-1, 0, rep],
    )
    # save plot
    # plt.savefig(os.path.join(pic_dir, "SVM_example.png"), dpi=300, bbox_inches="tight")


def plot_SVM(X, Y, coef, intercept):
    """
    plot the SVM hyperplane and the choice axis
    """
    # plot the SVM hyperplane

    z = lambda x, y: (-intercept - coef[0] * x - coef[1] * y) / coef[2]

    tmp = np.linspace(-10, 10, 30)
    x, y = np.meshgrid(tmp, tmp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X = X.T
    ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], "og", alpha=0.5)
    ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], "or", alpha=0.5)
    ax.plot_surface(x, y, z(x, y))

    # plot the choice axis
    orth_base = np.linspace(-10, 10, 100)
    ax.plot3D(
        orth_base * coef[0],
        orth_base * coef[1],
        orth_base * coef[2],
        color="black",
    )
    ax.view_init(8, -12)


def project_to_choice_axis(choice_axis, model_intercept, data):
    """
    project data to the choice axis
    """
    new_data = np.empty((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            new_data[i, j] = np.dot(data[:, i, j], choice_axis) - model_intercept
    return new_data


def project_all_data(
    all_act_mat_pca,
    all_coef_li,
    all_intercept_li,
    all_model_acc_li,
    total_rep=50,
    rerun_calc=False,
):
    """
    project all data to the choice axis
    """
    if rerun_calc:
        all_proj_li = []
        for coef, intercept, act_mat_pca, acc in zip(
            all_coef_li, all_intercept_li, all_act_mat_pca, all_model_acc_li
        ):
            model_proj_li = []
            for rep in tqdm(range(total_rep)):
                # use best performing of the 10 CV results as the choice axis
                best_model_idx = np.argmax(acc[:, rep])
                model_intercept = intercept[best_model_idx, 0, rep]
                choice_axis = coef[best_model_idx, :, rep]
                proj = project_to_choice_axis(
                    choice_axis, model_intercept, act_mat_pca[rep]
                )
                model_proj_li.append(proj)
            all_proj_li.append(model_proj_li)
        # save data
        with open(os.path.join(data_dir, "all_proj_li.pkl"), "wb") as f:
            dump(all_proj_li, f)
    else:
        with open(os.path.join(data_dir, "all_proj_li.pkl"), "rb") as f:
            all_proj_li = load(f)
    return all_proj_li


def calc_potential_t(pos_t, speed_t):
    unique_pos_t = np.unique(pos_t)
    potential_t = []
    for pos in unique_pos_t:
        idx = pos_t <= pos
        potential_tx = np.sum(speed_t[idx])
        potential_t.append(potential_tx)
    return np.array(potential_t)


def calculate_potential(proj, label):
    """
    Output:
        potential: list, len=#conditions, each element is a 2d array, shape=(#unique_pos, #time)
        positions: nested list, len=#conditions, each element is a 2d array, shape=(#unique_pos, #time)
        mean_act: 2d array, shape=(#conditions, # time)
    """
    cond = np.unique(label)
    potential = []
    positions = []
    mean_act = []
    for c in cond:
        cond_mean = np.mean(proj[label == c], axis=0)
        cond_arr = proj[label == c]
        dxdt = np.diff(cond_arr, axis=1) * cond_mean[1:]
        cond_potential = []
        cond_pos = []
        for t in range(dxdt.shape[1]):
            potential_t = calc_potential_t(cond_arr[:, t + 1], dxdt[:, t])
            cond_potential.append(potential_t)
            cond_pos.append(cond_arr[:, t + 1])
        cond_potential = np.stack(cond_potential, axis=1)
        cond_pos = np.stack(cond_pos, axis=1)

        potential.append(cond_potential)
        positions.append(cond_pos)
        mean_act.append(cond_mean)
    mean_act = np.stack(mean_act, axis=0)
    return potential, positions, mean_act


def calc_all_potential(all_proj_li, rerun_calc=False):
    """
    Output:
        all_potential_li: list, len=#models, each element is a list, len=#reps, each element is a list, len=#conditions, each element is a 2d array, shape=(#unique_pos, #time)
        all_unique_pos_li: list, len=#models, each element is a list, len=#reps, each element is a nested list, len=#conditions, each element is a 2d array, shape=(#unique_pos, #time)
        all_mean_act_li: list, len=#models, each element is a list, len=#reps, each element is 2d array, shape=(#conditions, # time)
    """
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
                _, _, label = load_sac_act(f_dirs[m_idx], rep, reload=False)
                potential, positions, mean_act = calculate_potential(proj, label)
                model_potential_li.append(potential)
                model_pos_li.append(positions)
                model_mean_act_li.append(mean_act)
            all_potential_li.append(model_potential_li)
            all_pos_li.append(model_pos_li)
            all_mean_act_li.append(model_mean_act_li)
        # save data
        with open(os.path.join(data_dir, "all_potential_li.pkl"), "wb") as f:
            dump(all_potential_li, f)
        with open(os.path.join(data_dir, "all_pos_li.pkl"), "wb") as f:
            dump(all_pos_li, f)
        with open(os.path.join(data_dir, "all_mean_act_li.pkl"), "wb") as f:
            dump(all_mean_act_li, f)
    else:
        with open(os.path.join(data_dir, "all_potential_li.pkl"), "rb") as f:
            all_potential_li = load(f)
        with open(os.path.join(data_dir, "all_pos_li.pkl"), "rb") as f:
            all_pos_li = load(f)
        with open(os.path.join(data_dir, "all_mean_act_li.pkl"), "rb") as f:
            all_mean_act_li = load(f)
    return all_potential_li, all_pos_li, all_mean_act_li


def plot_mean_potential_heatmap(
    avg_all_potential_agg, avg_mean_act, bins, model_fig_dir
):
    plt.figure(figsize=(12, 6))
    # find max and min of avg_all_potential_agg
    min_potential = np.inf
    max_potential = -np.inf
    for i in range(len(f_dirs)):
        max_potential = max(
            max_potential, np.nanmax(np.vstack(avg_all_potential_agg[i]))
        )
        min_potential = min(
            min_potential, np.nanmin(np.vstack(avg_all_potential_agg[i]))
        )

    for i in range(len(f_dirs)):
        plt.subplot(2, 2, i + 1)
        potential_agg = avg_all_potential_agg[i]
        plt.imshow(
            potential_agg,
            vmax=500,
            vmin=min_potential,
            aspect="auto",
            cmap="jet",
            origin="lower",
        )
        plt.vlines(25, 0, 60, color="white", linewidth=1, linestyle="--")
        plt.hlines(
            np.digitize(0, bins),
            0,
            50,
            color="white",
            linewidth=1,
            linestyle="--",
        )
        for j in range(avg_mean_act.shape[1]):
            plt.plot(np.digitize(avg_mean_act[i, j, :], bins) - 1, linewidth=2.5)
        plt.yticks(
            np.linspace(0, 60, 10),
            labels=np.round(np.linspace(bins[0], bins[-1], 10), 2),
        )
        plt.title(f_dirs[i].split("_")[-2])
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_fig_dir, "landscape_avg.png"),
        dpi=300,
        bbox_inches="tight",
    )
    return


def plot_model_potential_heatmap(
    rep, potential_li, unique_pos_li, mean_act_li, model_fig_dir, data_dir, f_dirs
):
    # get postion range and potential range
    min_potential = min_pos = np.inf
    max_potential = max_pos = -np.inf
    for i in range(len(potential_li)):
        max_pos = max(max_pos, np.max(np.vstack(unique_pos_li[i])))
        min_pos = min(min_pos, np.min(np.vstack(unique_pos_li[i])))
        max_potential = max(max_potential, np.max(np.vstack(potential_li[i])))
        min_potential = min(min_potential, np.min(np.vstack(potential_li[i])))

    pos_range = (min_pos, max_pos)
    potential_range = (min_potential, max_potential)

    plt.figure(figsize=(12, 6))
    rep_potential_agg = []
    for i in range(len(potential_li)):
        plt.subplot(2, 2, i + 1)
        potential_agg, bins = bin_potential(
            potential_li[i], unique_pos_li[i], pos_range
        )
        potential_agg = plot_single_heatmap(
            potential_agg,
            bins,
            mean_act_li[i],
            potential_range,
        )
        plt.title(f_dirs[i].split("_")[-2])
        rep_potential_agg.append(potential_agg)
    rep_potential_agg = np.stack(rep_potential_agg, axis=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_fig_dir, "landscape_%s.png" % str(rep)),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # save data
    with open(os.path.join(data_dir, "rep_potential_agg_%s.pkl" % str(rep)), "wb") as f:
        dump(rep_potential_agg, f)
    return rep_potential_agg


def plot_single_heatmap(potential_agg, bins, mean_act, potential_range, n_pos_bin=60):
    plt.imshow(
        potential_agg,
        vmax=max(potential_range),
        vmin=min(potential_range),
        aspect="auto",
        cmap="jet",
        origin="lower",
    )
    plt.vlines(25, 0, n_pos_bin, color="white", linewidth=1, linestyle="--")
    plt.hlines(
        np.digitize(0, bins),
        0,
        50,
        color="white",
        linewidth=1,
        linestyle="--",
    )
    for i in range(len(mean_act)):
        plt.plot(np.digitize(mean_act[i], bins) - 1, linewidth=2.5)
    plt.yticks(
        np.linspace(0, n_pos_bin, 10),
        labels=np.round(np.linspace(bins[0], bins[-1], 10), 2),
    )
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.colorbar()
    return potential_agg


def bin_potential(potential, positions, pos_range, n_pos_bin=60):
    potential_mat = np.vstack(potential)
    pos_mat = np.vstack(positions)
    binned_pos, bins = bin_pos(pos_mat, n_pos_bin, pos_range)
    potential_agg = aggregate_potential(potential_mat, binned_pos, bins)
    return potential_agg, bins


def aggregate_potential(potential, binned_pos, bins):
    """
    aggregate potential to bins
    """
    unique_pos = np.unique(binned_pos)
    agg_potential = np.nan * np.ones((len(bins), potential.shape[1]))
    for pos in unique_pos:
        for t in range(potential.shape[1]):
            idx = binned_pos[:, t] == pos
            if np.sum(idx) == 0:
                continue
            agg_potential[np.digitize(pos, bins) - 1, t] = np.mean(
                potential[:, t][idx], axis=0
            )
    return agg_potential


def bin_pos(pos, n_pos_bin, pos_range):
    # flat_pos = np.ravel(np.hstack(pos))
    # min_pos = np.min(flat_pos)
    # max_pos = np.max(flat_pos)
    bins = np.linspace(min(pos_range), max(pos_range), n_pos_bin)
    pos_bins = np.digitize(pos, bins)
    binned_pos = (pos_bins - 1) * (bins[1] - bins[0]) + bins[0]
    return binned_pos, bins


if __name__ == "__main__":
    main()
