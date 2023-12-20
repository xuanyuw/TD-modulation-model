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
from pickle import load, dump
from tqdm import tqdm


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
proj_data_dir = os.path.join("new_dyn_analysis", "data")
data_dir = os.path.join("new_dyn_analysis", "data")
# data_dir = os.path.join("new_dyn_analysis", "data", "50bins")

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

    # find max and min in all_pos_li, define spatial bins
    min_pos = np.inf
    max_pos = -np.inf
    for i in range(len(all_proj_li)):
        for j in range(len(all_proj_li[i])):
            min_pos = min(min_pos, np.min(np.vstack(all_proj_li[i][j])))
            max_pos = max(max_pos, np.max(np.vstack(all_proj_li[i][j])))
    bins = np.linspace(min_pos, max_pos, 100)

    # plot_proj_example(all_proj_li, 3)
    all_potential_dict, all_mean_act_dict = calc_all_potential(
        all_proj_li, bins, rerun_calc=False
    )
    model_fig_dir = os.path.join(pic_dir, "temp_landscape_50bins")
    if not os.path.exists(model_fig_dir):
        os.makedirs(model_fig_dir)
        # for rep in tqdm(range(total_rep)):
        #     if replot_single_model:
        #         rep_potential_li = [li[rep] for li in all_potential_li]
        #         rep_mean_act_li = [li[rep] for li in all_mean_act_li]
        #         plot_model_potential_heatmap(
        #             rep,
        #             rep_potential_li,
        #             rep_mean_act_li,
        #             model_fig_dir,
        #             data_dir,
        #             f_dirs,
        #             bins,
        #         )
        # plot mean potential heatmap
        # find max and min of avg_all_potential_agg
    avg_all_potential_agg = np.nanmean(np.array(all_potential_dict["all"]), axis=1)
    min_potential = np.nanmin(avg_all_potential_agg)
    max_potential = np.nanmax(avg_all_potential_agg)

    for k in list(all_potential_dict.keys()):
        plot_mean_potential_heatmap(
            all_potential_dict[k],
            all_mean_act_dict[k],
            bins,
            model_fig_dir,
            k,
            (min_potential, max_potential),
        )


################################# data projection #################################
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
        with open(os.path.join(proj_data_dir, "all_proj_li.pkl"), "rb") as f:
            all_proj_li = load(f)
    return all_proj_li


################################# potential calculation #################################


def calc_exp_speed(binned_pos, speed, n_bins):
    exp_speed = np.ones((n_bins, speed.shape[1])) * np.nan
    for t in range(speed.shape[1]):
        for pos in np.unique(binned_pos):
            idx = binned_pos[:, t] == pos
            if np.sum(idx) == 0:
                continue
            exp_speed[pos, t] = np.nanmean(speed[idx, t])
    return exp_speed


def calc_cond_potential(dxdt, bins, c):
    potential = np.nan * np.ones(dxdt.shape)
    z_idx = np.digitize(0, bins) - 1
    all_potential_ref = []
    for t in range(dxdt.shape[1]):
        potential_ref = np.nansum(dxdt[:z_idx, t]) * (-1)
        all_potential_ref.append(potential_ref)
        for p in bins:
            p_idx = np.digitize(p, bins) - 1
            select_idx = bins < p
            if np.isnan(dxdt[select_idx, t]).all():
                potential_tx = np.nan
            else:
                potential_tx = np.nansum(dxdt[select_idx, t]) * (-1) - potential_ref
            potential[p_idx, t] = potential_tx

    if c == 0:
        potential[z_idx + 1 :, :] = np.nan
    else:
        potential[: z_idx - 1, :] = np.nan
    potential = potential * (1 - np.isnan(dxdt))
    potential[potential == 0] = np.nan
    potential[z_idx, :] = 0
    return potential


def calc_model_potential(proj, label, bins):
    """
    Output:
        potential: 2d array, shape=(#bins, #time)
        mean_act: 2d array, shape=(#conditions, # time)
    """
    cond = np.unique(label)
    mean_act = []
    binned_proj = np.digitize(proj, bins) - 1
    potential = []
    #
    all_exp_speed = np.nan * np.ones((len(bins), proj.shape[1] - 1))
    for c in cond:
        binned_cond_proj = binned_proj[label == c]
        cond_mean = np.mean(binned_cond_proj, axis=0)

        x_diff = np.diff(binned_cond_proj, axis=1)
        exp_speed = calc_exp_speed(binned_cond_proj, x_diff, len(bins))

        z_idx = np.digitize(0, bins) - 1
        if c == 0:
            all_exp_speed[:z_idx, :] = exp_speed[:z_idx, :]
        else:
            all_exp_speed[z_idx:, :] = exp_speed[z_idx:, :]

        cond_potential = calc_cond_potential(exp_speed, bins, c)
        potential.append(cond_potential)

        mean_act.append(cond_mean)
    mean_act = np.stack(mean_act, axis=0)

    return potential, mean_act


def merge_choice_potential(potential_li, bins):
    potential = np.nan * np.ones(potential_li[0].shape)
    z_idx = np.digitize(0, bins) - 1
    for c in range(len(potential_li)):
        cond_potential = potential_li[c]
        if c == 0:
            potential[:z_idx, :] = cond_potential[:z_idx, :]
        else:
            potential[z_idx:, :] = cond_potential[z_idx:, :]
    return potential


def calc_all_potential(all_proj_li, bins, rerun_calc=True):
    """
    Output:
        all_potential_li: list, len=#models, each element is a list, len=#reps, each element is a list, len=#conditions, each element is a 2d array, shape=(#unique_pos, #time)
        all_unique_pos_li: list, len=#models, each element is a list, len=#reps, each element is a nested list, len=#conditions, each element is a 2d array, shape=(#unique_pos, #time)
        all_mean_act_li: list, len=#models, each element is a list, len=#reps, each element is 2d array, shape=(#conditions, # time)
    """
    if rerun_calc:
        all_potential_dict = {"all": [], "H": [], "M": [], "L": [], "Z": []}
        all_mean_act_dict = {"all": [], "H": [], "M": [], "L": [], "Z": []}
        for m_idx, model_proj_li in enumerate(all_proj_li):
            model_potential_dict = {"all": [], "H": [], "M": [], "L": [], "Z": []}
            model_mean_act_dict = {"all": [], "H": [], "M": [], "L": [], "Z": []}
            for rep in tqdm(range(len(model_proj_li))):
                proj = model_proj_li[rep]
                _, _, label = load_sac_act(f_dirs[m_idx], rep, reload=False)
                coh_idx = load_coh_dict(f_dirs[m_idx], rep)
                potential, mean_act = calc_model_potential(proj, label, bins)
                potential = merge_choice_potential(potential, bins)
                model_potential_dict["all"].append(potential)
                model_mean_act_dict["all"].append(mean_act)
                for coh in list(coh_idx.keys()):
                    coh_label = label[coh_idx[coh]]
                    coh_proj = proj[coh_idx[coh]]
                    coh_potential, coh_mean_act = calc_model_potential(
                        coh_proj, coh_label, bins
                    )
                    coh_potential = merge_choice_potential(coh_potential, bins)
                    model_potential_dict[coh].append(coh_potential)
                    model_mean_act_dict[coh].append(coh_mean_act)

            for k in all_potential_dict.keys():
                all_potential_dict[k].append(model_potential_dict[k])
                all_mean_act_dict[k].append(model_mean_act_dict[k])

        # save data
        with open(os.path.join(data_dir, "all_potential_dict.pkl"), "wb") as f:
            dump(all_potential_dict, f)
        with open(os.path.join(data_dir, "all_mean_act_dict.pkl"), "wb") as f:
            dump(all_mean_act_dict, f)
    else:
        with open(os.path.join(data_dir, "all_potential_dict.pkl"), "rb") as f:
            all_potential_dict = load(f)
        with open(os.path.join(data_dir, "all_mean_act_dict.pkl"), "rb") as f:
            all_mean_act_dict = load(f)
    return all_potential_dict, all_mean_act_dict


def load_coh_dict(f_dir, rep):
    n = SimpleNamespace(
        **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (0.02, rep))
    )
    coh_dict = find_coh_idx(n.stim_level)
    return coh_dict


################### plotting functions ####################
def min_max_scaler(X, min, max):
    # X_std = (X - min) / (max - min)
    # return X_std * (1 - (-1)) + (-1)  # scale to (-1, 1)
    return X / np.abs(min)


# def min_max_scaler(X):
#     X_std = X / np.abs(np.nanmax(X))
#     return X_std


def plot_mean_potential_heatmap(
    all_potential_agg,
    all_mean_act,
    bins,
    model_fig_dir,
    coh,
    potential_range,
    normalize=False,
):
    if normalize:
        # min-max normalize potential and mean_act
        for j in range(len(all_potential_agg[0])):
            min_p = np.nanmin(np.array(all_potential_agg)[:, j, :, :])
            max_p = np.nanmax(np.array(all_potential_agg)[:, j, :, :])
            for i in range(len(all_potential_agg)):
                all_potential_agg[i][j] = min_max_scaler(
                    all_potential_agg[i][j], min_p, max_p
                )
                # all_potential_agg[i][j] = min_max_scaler(all_potential_agg[i][j])
                # all_mean_act[i][j] = min_max_scaler(all_mean_act[i][j])

    avg_all_potential_agg = np.nanmean(np.array(all_potential_agg), axis=1)
    avg_mean_act = np.nanmean(np.array(all_mean_act), axis=1)

    min_potential = min(potential_range)
    max_potential = max(potential_range)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i in range(len(f_dirs)):
        ax1 = axes[idx[i][0], idx[i][1]]
        potential_agg = avg_all_potential_agg[i]
        s = ax1.imshow(
            potential_agg,
            vmax=max_potential,
            vmin=min_potential,
            aspect="auto",
            origin="lower",
        )
        plt.colorbar(s, ax=ax1)

        # if normalize:
        # ax2 = ax1.twinx()
        # ax2.plot(avg_mean_act[i, :, :].T, linewidth=2.5)
        # ax2.set_ylabel("Normalzied Position")

        # else:
        # ax1.plot(avg_mean_act[i, :, :].T - 1, linewidth=2.5)
        ax1.plot(avg_mean_act[i, :, :].T, linewidth=2.5)

        # # set symmetrical ylim
        # yl = ax1.get_ylim()
        # y_abs_max = max(np.abs(yl))
        # ax1.set_ylim((-y_abs_max, y_abs_max))

        # set int ytick
        # TODO: only applicable to 100 bins
        ax1.set_ylim(-10, 100)
        ax1.set_yticks(
            np.linspace(-10, 100, 9),
            labels=np.round(np.linspace(-20, 20, 9)).astype(int),
        )
        # ax1.set_yticks(
        #     np.linspace(-10, 100, 9),
        #     labels=np.round(np.linspace(-19, 19, 9)).astype(int),
        # )
        ax1.set_title(f_dirs[i].split("_")[-2])
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Position")

        ax1.vlines(25, 0, 60, color="white", linewidth=1, linestyle="--")
        ax1.hlines(
            np.digitize(0, bins),
            0,
            50,
            color="white",
            linewidth=1,
            linestyle="--",
        )
    plt.tight_layout()
    # plt.savefig(
    #     os.path.join(model_fig_dir, "landscape_avg.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    plt.savefig(
        os.path.join(model_fig_dir, "landscape_avg_%s.pdf" % coh),
        format="pdf",
        bbox_inches="tight",
    )
    return


def plot_model_potential_heatmap(
    rep, potential_li, mean_act_li, model_fig_dir, data_dir, f_dirs, bins
):
    # get postion range and potential range
    max_potential = np.nanmax(np.array(potential_li))
    min_potential = np.nanmin(np.array(potential_li))
    potential_range = (min_potential, max_potential)

    plt.figure(figsize=(12, 6))
    rep_potential_agg = []
    for i in range(len(potential_li)):
        plt.subplot(2, 2, i + 1)
        potential_agg = plot_single_heatmap(
            potential_li[i],
            bins,
            mean_act_li[i],
            potential_range,
            n_pos_bin=len(bins),
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


def plot_single_heatmap(potential_agg, bins, mean_act, potential_range, n_pos_bin):
    plt.imshow(
        potential_agg,
        vmax=max(potential_range),
        vmin=min(potential_range),
        aspect="auto",
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
        plt.plot(mean_act[i], linewidth=2.5)
    plt.yticks(
        np.linspace(0, n_pos_bin, 10),
        labels=np.round(np.linspace(bins[0], bins[-1], 10), 2),
    )
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.colorbar()
    return potential_agg


############################### SVM illustration ###############################


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


if __name__ == "__main__":
    main()
