import sys
import os

import brainpy as bp
import brainpy.math as bm

from types import SimpleNamespace
from utils import *
from model.model import Model
from model.calc_params import par, update_parameters

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


lr = 0.02
# rep = 0
total_rep = 50
# plot_coh = "H"


f_dirs = [
    # "test_output_full_model",
    # "test_output_noFeedback_model",
    "cutSpec_model",
    "cutNonspec_model",
]
all_cohs = ["H", "M", "L", "Z"]
rerun_calculation = False

plot_dir = "slow_point_2D_plots"
FIX_INIT_TIME = 13
TARGET_INIT_TIME = 35
STIM_INIT_TIME = 58


def main(f_dir, plot_coh, rep, fp_run_counter):
    model_type = f_dir.split("_")[-2]
    save_dir = os.path.join(plot_dir, model_type, "rep%d" % rep)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
    correct_idx = np.where(n.correct_idx)[0]
    incorrect_idx = np.where(~n.correct_idx)[0]

    pca_all = construct_PC_space(normalized_h, stim_dir, reshaped_out)

    activity_dict_all = [normalized_h[:, i, :] for i in range(normalized_h.shape[1])]

    Z_idx = np.where(np.logical_and(np.array(n.stim_level) == b"Z", n.correct_idx))[0]
    L_idx = np.where(np.logical_and(np.array(n.stim_level) == b"L", n.correct_idx))[0]
    M_idx = np.where(np.logical_and(np.array(n.stim_level) == b"M", n.correct_idx))[0]
    H_idx = np.where(np.logical_and(np.array(n.stim_level) == b"H", n.correct_idx))[0]
    non_Z_idx = np.where(np.logical_and(np.array(n.stim_level) != b"Z", n.correct_idx))[
        0
    ]
    targ_arrange = recover_targ_loc(n.desired_out, n.stim_dir)[-1, :]

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

    print("-------------------------------")
    print("Running/Loading Fixed Points for model %s rep %d..." % (model_type, rep))
    if rerun_calculation and fp_run_counter == 0:
        fp_gl_dict, fp_rl_dict = get_fixed_points(
            n, targ_arrange, save_dir, rerun_calculation
        )
        fp_run_counter += 1
    else:
        # avoid repeating the same calculation
        fp_gl_dict, fp_rl_dict = get_fixed_points(n, targ_arrange, save_dir, False)
    transform_and_save_fp(pca_all, fp_gl_dict, fp_rl_dict, save_dir)

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

    plt.savefig(os.path.join(save_dir, plt_fn_png))
    plt.savefig(os.path.join(save_dir, plt_fn_pdf))

    plt.close("all")


def construct_PC_space(normalized_h, stim_dir, reshaped_out):
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

    pca_all = PCA(n_components=3)
    pca_all.fit(prep_h_mat)

    return pca_all


def get_fixed_points(n, targ_arrange, save_dir, rerun_calculation):
    if rerun_calculation:
        model = Model(par, n.neural_input, train=False)
        model.slow_point_update = True
        model.reset_batch()

        gl_h_idx = np.where(
            np.logical_and(
                np.logical_and(np.array(n.stim_level) == b"H", n.correct_idx),
                targ_arrange == 0,
            )
        )[0]
        fix_gl_fp_finder = build_finder(FIX_INIT_TIME, gl_h_idx, n, model)
        target_gl_fp_finder = build_finder(TARGET_INIT_TIME, gl_h_idx, n, model)
        stim_gl_fp_finder = build_finder(STIM_INIT_TIME, gl_h_idx, n, model)

        rl_h_idx = np.where(
            np.logical_and(
                np.logical_and(np.array(n.stim_level) == b"H", n.correct_idx),
                targ_arrange == 1,
            )
        )[0]

        fix_rl_fp_finder = build_finder(FIX_INIT_TIME, rl_h_idx, n, model)
        target_rl_fp_finder = build_finder(TARGET_INIT_TIME, rl_h_idx, n, model)
        stim_rl_fp_finder = build_finder(STIM_INIT_TIME, rl_h_idx, n, model)

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

        finder_dict = {
            "fix_gl": {
                "h": fix_gl_fp_finder.fixed_points["h"],
                "syn_x": fix_gl_fp_finder.fixed_points["syn_x"],
                "syn_u": fix_gl_fp_finder.fixed_points["syn_u"],
            },
            "target_gl": {
                "h": target_gl_fp_finder.fixed_points["h"],
                "syn_x": target_gl_fp_finder.fixed_points["syn_x"],
                "syn_u": target_gl_fp_finder.fixed_points["syn_u"],
            },
            "stim_gl": {
                "h": stim_gl_fp_finder.fixed_points["h"],
                "syn_x": stim_gl_fp_finder.fixed_points["syn_x"],
                "syn_u": stim_gl_fp_finder.fixed_points["syn_u"],
            },
            "fix_rl": {
                "h": fix_rl_fp_finder.fixed_points["h"],
                "syn_x": fix_rl_fp_finder.fixed_points["syn_x"],
                "syn_u": fix_rl_fp_finder.fixed_points["syn_u"],
            },
            "target_rl": {
                "h": target_rl_fp_finder.fixed_points["h"],
                "syn_x": target_rl_fp_finder.fixed_points["syn_x"],
                "syn_u": target_rl_fp_finder.fixed_points["syn_u"],
            },
            "stim_rl": {
                "h": stim_rl_fp_finder.fixed_points["h"],
                "syn_x": stim_rl_fp_finder.fixed_points["syn_x"],
                "syn_u": stim_rl_fp_finder.fixed_points["syn_u"],
            },
        }
        with open(os.path.join(save_dir, "fp_dicts.pth"), "wb") as f:
            np.save(f, finder_dict)
    else:
        with open(os.path.join(save_dir, "fp_dicts.pth"), "rb") as f:
            finder_dict = np.load(f, allow_pickle=True).item()
        fp_gl_dict = {
            "fix": finder_dict["fix_gl"]["h"],
            "target": finder_dict["target_gl"]["h"],
            "stim": finder_dict["stim_gl"]["h"],
        }
        fp_rl_dict = {
            "fix": finder_dict["fix_rl"]["h"],
            "target": finder_dict["target_rl"]["h"],
            "stim": finder_dict["stim_rl"]["h"],
        }

    return fp_gl_dict, fp_rl_dict


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


def transform_fixed_points(pca, fp_dict):
    fix_fps = pca.transform(fp_dict["fix"])
    target_fps = pca.transform(fp_dict["target"])
    stim_fps = pca.transform(fp_dict["stim"])
    return fix_fps, target_fps, stim_fps


def transform_and_save_fp(pca, fp_gl_dict, fp_rl_dict, save_dir):
    fix_fps_gl, target_fps_gl, stim_fps_gl = transform_fixed_points(pca, fp_gl_dict)
    fix_fps_rl, target_fps_rl, stim_fps_rl = transform_fixed_points(pca, fp_rl_dict)
    fp_transformed = {
        "fix_gl": fix_fps_gl,
        "target_gl": target_fps_gl,
        "stim_gl": stim_fps_gl,
        "fix_rl": fix_fps_rl,
        "target_rl": target_fps_rl,
        "stim_rl": stim_fps_rl,
    }
    with open(os.path.join(save_dir, "h_fp_transformed.pth"), "wb") as f:
        np.save(f, fp_transformed)


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

    fix_fps1, target_fps1, stim_fps1 = transform_fixed_points(pca, fp_dict1)
    fix_fps2, target_fps2, stim_fps2 = transform_fixed_points(pca, fp_dict2)

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
    # for rep in range(total_rep):
    for rep in range(0, 39):
        fp_run_counter = 0
        for c in all_cohs:
            main(f_dir, c, rep, fp_run_counter)
            fp_run_counter += 1
