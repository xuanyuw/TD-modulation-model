import sys
import os

import brainpy as bp
import brainpy.math as bm

from types import SimpleNamespace
from utils import *
from model import Model
from calc_params import par, update_parameters

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# f_dir = "test_output_full_model"
lr = 0.02
rep = 0
# plot_coh = "H"


f_dirs = [
    # "test_output_full_model",
    # "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
    "test_output_noFeedback_model"
    # "cutSpec_model",
    # "cutNonspec_model",
]
all_cohs = ["H", "M", "L", "Z"]

plot_dir = "slow_point_2D_plots"


def main(f_dir, plot_coh):
    model_type = f_dir.split("_")[-2]
    update_parameters(
        {
            "rep": rep,
            "save_fn": "model_results_%d_lr%f.pkl" % (rep, lr),
            "batch_size": par["test_batch_size"],
            "num_iterations": par["num_test_iterations"],
            "coherence_levels": par["test_coherence_levels"],
            "weight_fn": "weight_%d.pth" % (rep),
            "learning_rate": lr,
            "save_test_out": False,
        }
    )
    n = SimpleNamespace(
        **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
    )
    normalized_h = min_max_normalize(n.h)

    stim_dir = n.stim_dir
    reshaped_out = n.desired_out[-1, :, :]
    out = [0 if reshaped_out[i, 0] == 1 else 1 for i in range(reshaped_out.shape[0])]
    cond1_idx = np.logical_and(stim_dir == 135, np.array(out) == 0)
    cond2_idx = np.logical_and(stim_dir == 135, np.array(out) == 1)
    cond3_idx = np.logical_and(stim_dir == 315, np.array(out) == 0)
    cond4_idx = np.logical_and(stim_dir == 315, np.array(out) == 1)

    cond1_h = np.mean(normalized_h[:, cond1_idx, :], axis=1)
    cond2_h = np.mean(normalized_h[:, cond2_idx, :], axis=1)
    cond3_h = np.mean(normalized_h[:, cond3_idx, :], axis=1)
    cond4_h = np.mean(normalized_h[:, cond4_idx, :], axis=1)

    prep_h_mat = np.concatenate((cond1_h, cond2_h, cond3_h, cond4_h), axis=0)

    correct_idx = np.where(n.correct_idx)[0]
    incorrect_idx = np.where(~n.correct_idx)[0]

    pca_all = PCA(n_components=3)
    pca_all.fit(prep_h_mat)

    activity_dict_all = [normalized_h[:, i, :] for i in range(normalized_h.shape[1])]

    Z_idx = np.where(np.logical_and(np.array(n.stim_level) == b"Z", n.correct_idx))[0]
    L_idx = np.where(np.logical_and(np.array(n.stim_level) == b"L", n.correct_idx))[0]
    M_idx = np.where(np.logical_and(np.array(n.stim_level) == b"M", n.correct_idx))[0]
    H_idx = np.where(np.logical_and(np.array(n.stim_level) == b"H", n.correct_idx))[0]
    non_Z_idx = np.where(np.logical_and(np.array(n.stim_level) != b"Z", n.correct_idx))[
        0
    ]
    targ_arrange = recover_targ_loc(n.desired_out, n.stim_dir)[-1, :]

    model = Model(par, n.neural_input, train=False)
    model.slow_point_update = True
    model.reset_batch()

    gl_h_idx = np.where(
        np.logical_and(
            np.logical_and(np.array(n.stim_level) == b"H", n.correct_idx),
            targ_arrange == 0,
        )
    )[0]
    fix_gl_fp_finder = build_finder(13, gl_h_idx, n, model)
    target_gl_fp_finder = build_finder(35, gl_h_idx, n, model)
    stim_gl_fp_finder = build_finder(58, gl_h_idx, n, model)

    rl_h_idx = np.where(
        np.logical_and(
            np.logical_and(np.array(n.stim_level) == b"H", n.correct_idx),
            targ_arrange == 1,
        )
    )[0]

    fix_rl_fp_finder = build_finder(13, rl_h_idx, n, model)
    target_rl_fp_finder = build_finder(35, rl_h_idx, n, model)
    stim_rl_fp_finder = build_finder(58, rl_h_idx, n, model)

    fp_gl_dict = {
        "fix": fix_gl_fp_finder.fixed_points["h"],
        "target": target_gl_fp_finder.fixed_points["h"],
        "stim": stim_gl_fp_finder.fixed_points["h"],
    }
    fp_rl_dict = {
        "fix": fix_rl_fp_finder.fixed_points["h"],
        "target": target_rl_fp_finder.fixed_points["h"],
        "stim": stim_rl_fp_finder.fixed_points["h"],
    }

    # color_dict_out = {0: "orange", 1: "blue"}
    # color_dict_stim = {135: "green", 315: "red"}
    color_dict_choice_c = {0: "green", 1: "red"}

    # cond_comb = []# cond comb: 0 = targ green left, saccade left, 1 = targ green left, saccade right, 2 = targ red left, saccade left, 3 = targ red left, saccade right
    # for i in range(len(targ_arrange)):
    #     if targ_arrange[i] == 0 and out[i] == 0:
    #         cond_comb.append(0)
    #     elif targ_arrange[i] == 0 and out[i] == 1:
    #         cond_comb.append(1)
    #     elif targ_arrange[i] == 1 and out[i] == 0:
    #         cond_comb.append(2)
    #     elif targ_arrange[i] == 1 and out[i] == 1:
    #         cond_comb.append(3)

    cond_comb = (
        []
    )  # cond comb: 0 = targ green left, stim green, 1 = targ green left, stim red, 2 = targ red left, stim green, 3 = targ red left, stim red
    for i in range(len(targ_arrange)):
        if targ_arrange[i] == 0 and stim_dir[i] == 135:
            cond_comb.append(0)
        elif targ_arrange[i] == 0 and stim_dir[i] == 315:
            cond_comb.append(1)
        elif targ_arrange[i] == 1 and stim_dir[i] == 135:
            cond_comb.append(2)
        elif targ_arrange[i] == 1 and stim_dir[i] == 315:
            cond_comb.append(3)

    color_dict_targ_out_comb = {0: "#77945C", 1: "#1FA808", 2: "#B25F4A", 3: "#ED2938"}
    color_dict_targ = {0: "#77945C", 1: "#B25F4A"}

    if plot_coh == "H":
        plt_idx = H_idx
    elif plot_coh == "M":
        plt_idx = M_idx
    elif plot_coh == "L":
        plt_idx = L_idx
    elif plot_coh == "Z":
        plt_idx = Z_idx
    elif plot_coh == "correct":
        plt_idx = correct_idx
    elif plot_coh == "incorrect":
        plt_idx = incorrect_idx

    plt_fn_png = "".join([model_type, "_", plot_coh, "_2D_subplots.png"])
    plt_fn_pdf = "".join([model_type, "_", plot_coh, "_2D_subplots.pdf"])

    plot2d_subplots_with_fps(
        activity_dict_all,
        fp_gl_dict,
        fp_rl_dict,
        pca_all,
        plt_idx,
        cond_comb,
        color_dict_targ_out_comb,
        "".join([model_type, "_", plot_coh]),
    )

    if not os.path.exists(os.path.join(plot_dir, model_type)):
        os.makedirs(os.path.join(plot_dir, model_type))

    plt.savefig(os.path.join(plot_dir, model_type, plt_fn_png))
    plt.savefig(os.path.join(plot_dir, model_type, plt_fn_pdf))


def build_finder(time_period, idx, n, model):
    fp_h_init = n.h[time_period, idx, :]
    fp_synx_init = n.syn_x[time_period, idx, :]
    fp_synu_init = n.syn_u[time_period, idx, :]
    fp_candidates = {"h": fp_h_init, "syn_x": fp_synx_init, "syn_u": fp_synu_init}
    finder = bp.analysis.SlowPointFinder(
        model,
        args=(n.neural_input[time_period, 0, :],),
        target_vars={"h": model.h, "syn_x": model.syn_x, "syn_u": model.syn_u},
    )
    finder.find_fps_with_gd_method(
        candidates=fp_candidates, num_batch=500, tolerance=5e-6
    )
    return finder


def plot2d_subplots_with_fps(
    activity_dict,
    fp_dict1,
    fp_dict2,
    pca,
    indices,
    ground_truth_arr,
    color_dict,
    title,
    color_dict_targ=None,
    fix_points_only=False,
):
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    fix_fps1 = pca.transform(fp_dict1["fix"])
    target_fps1 = pca.transform(fp_dict1["target"])
    stim_fps1 = pca.transform(fp_dict1["stim"])

    fix_fps2 = pca.transform(fp_dict2["fix"])
    target_fps2 = pca.transform(fp_dict2["target"])
    stim_fps2 = pca.transform(fp_dict2["stim"])
    if not fix_points_only:
        for i in indices[:100]:
            activity_pc = pca.transform(activity_dict[i])
            ground_truth = ground_truth_arr[i]
            color = color_dict[ground_truth]
            if color_dict_targ is not None:
                color_targ = (
                    list(color_dict_targ.values())[0]
                    if ground_truth == list(color_dict_targ.keys())[0]
                    else list(color_dict_targ.values())[1]
                )
            else:
                color_targ = color
            ax1.plot(
                activity_pc[:25, 0],
                activity_pc[:25, 1],
                "-",
                color="gray",
                alpha=0.2,
            )
            ax2.plot(
                activity_pc[:25, 0],
                activity_pc[:25, 2],
                "-",
                color="gray",
                alpha=0.2,
            )
            ax3.plot(
                activity_pc[:25, 1],
                activity_pc[:25, 2],
                "-",
                color="gray",
                alpha=0.2,
            )

            ax1.plot(
                activity_pc[25:45, 0],
                activity_pc[25:45, 1],
                "-",
                color=color_targ,
                alpha=0.2,
            )
            ax2.plot(
                activity_pc[25:45, 0],
                activity_pc[25:45, 2],
                "-",
                color=color_targ,
                alpha=0.2,
            )
            ax3.plot(
                activity_pc[25:45, 1],
                activity_pc[25:45, 2],
                "-",
                color=color_targ,
                alpha=0.2,
            )

            ax1.plot(
                activity_pc[45:, 0],
                activity_pc[45:, 1],
                "-",
                color=color,
                alpha=0.4,
            )
            ax2.plot(
                activity_pc[45:, 0],
                activity_pc[45:, 2],
                "-",
                color=color,
                alpha=0.4,
            )
            ax3.plot(
                activity_pc[45:, 1],
                activity_pc[45:, 2],
                "-",
                color=color,
                alpha=0.4,
            )

    ax1.plot(fix_fps1[:, 0], fix_fps1[:, 1], "x", color="#DBBD23")
    ax2.plot(fix_fps1[:, 0], fix_fps1[:, 2], "x", color="#DBBD23")
    ax3.plot(fix_fps1[:, 1], fix_fps1[:, 2], "x", color="#DBBD23")

    ax1.plot(fix_fps2[:, 0], fix_fps2[:, 1], "x", color="#DBBD23")
    ax2.plot(fix_fps2[:, 0], fix_fps2[:, 2], "x", color="#DBBD23")
    ax3.plot(fix_fps2[:, 1], fix_fps2[:, 2], "x", color="#DBBD23")

    ax1.plot(
        target_fps1[:, 0],
        target_fps1[:, 1],
        "x",
        color="#57B75A",
        alpha=0.5,
    )
    ax2.plot(
        target_fps1[:, 0],
        target_fps1[:, 2],
        "x",
        color="#57B75A",
        alpha=0.5,
    )
    ax3.plot(
        target_fps1[:, 1],
        target_fps1[:, 2],
        "x",
        color="#57B75A",
        alpha=0.5,
    )

    ax1.plot(
        target_fps2[:, 0],
        target_fps2[:, 1],
        "x",
        color="#D57617",
        alpha=0.5,
    )
    ax2.plot(
        target_fps2[:, 0],
        target_fps2[:, 2],
        "x",
        color="#D57617",
        alpha=0.5,
    )
    ax3.plot(
        target_fps2[:, 1],
        target_fps2[:, 2],
        "x",
        color="#D57617",
        alpha=0.5,
    )

    ax1.plot(stim_fps1[:, 0], stim_fps1[:, 1], "x", color="#2F593E")
    ax2.plot(stim_fps1[:, 0], stim_fps1[:, 2], "x", color="#2F593E")
    ax3.plot(stim_fps1[:, 1], stim_fps1[:, 2], "x", color="#2F593E")
    ax1.plot(stim_fps2[:, 0], stim_fps2[:, 1], "x", color="#932525")
    ax2.plot(stim_fps2[:, 0], stim_fps2[:, 2], "x", color="#932525")
    ax3.plot(stim_fps2[:, 1], stim_fps2[:, 2], "x", color="#932525")

    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC3")
    ax3.set_xlabel("PC2")
    ax3.set_ylabel("PC3")

    plt.suptitle(title)


for f_dir in f_dirs:
    for c in all_cohs:
        main(f_dir, c)
