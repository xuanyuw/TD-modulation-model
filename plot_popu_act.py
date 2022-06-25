import numpy as np
import tables
import matplotlib.pyplot as plt
import os


f_dir = "fix_inout_low_rf_conn_model"
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
    # plot population neural activity
    normalized_h = min_max_normalize(h)

    plot_dir_selectivity(
        normalized_h,
        (0, 32),
        (80, 112),
        y,
        desired_out,
        stim_level,
        stim_dir,
        "Motion_Excitatory_Direction_Selectivity",
        True,
    )
    plot_dir_selectivity(
        normalized_h,
        (32, 80),
        (112, 160),
        y,
        desired_out,
        stim_level,
        stim_dir,
        "Target_Excitatory_Direction_Selectivity",
        True,
    )
    plot_dir_selectivity(
        normalized_h,
        (160, 168),
        (180, 188),
        y,
        desired_out,
        stim_level,
        stim_dir,
        "Motion_Inhibitory_Direction_Selectivity",
        True,
    )
    plot_dir_selectivity(
        normalized_h,
        (168, 180),
        (188, 200),
        y,
        desired_out,
        stim_level,
        stim_dir,
        "Target_Inhibitory_Direction_Selectivity",
        True,
    )

    plot_sac_selectivity(
        normalized_h,
        (0, 32),
        (80, 112),
        y,
        desired_out,
        stim_level,
        "Motion_Excitatory_Saccade_Selectivity",
        True,
    )
    plot_sac_selectivity(
        normalized_h,
        (32, 80),
        (112, 160),
        y,
        desired_out,
        stim_level,
        "Target_Excitatory_Saccade_Selectivity",
        True,
    )
    plot_sac_selectivity(
        normalized_h,
        (160, 168),
        (180, 188),
        y,
        desired_out,
        stim_level,
        "Motion_Inhibitory_Saccade_Selectivity",
        True,
    )
    plot_sac_selectivity(
        normalized_h,
        (168, 180),
        (188, 200),
        y,
        desired_out,
        stim_level,
        "Target_Inhibitory_Saccade_Selectivity",
        True,
    )


def relu(input):
    return input * (input > 0)


def get_max_iter(table):
    all_iters = []
    for row in table:
        iter_num = int(row.name.split("iter")[1])
        if iter_num not in all_iters:
            all_iters.append(iter_num)
        else:
            break
    max_iter = max(all_iters)
    return max_iter


def find_coh_idx(stim_level):
    H_idx = np.array(stim_level) == b"H"
    M_idx = np.array(stim_level) == b"M"
    L_idx = np.array(stim_level) == b"L"
    Z_idx = np.array(stim_level) == b"Z"
    return H_idx, M_idx, L_idx, Z_idx


def find_sac_idx(y, m1):
    choice = np.argmax(y, 2)
    if m1:
        contra_idx = choice == 0
        ipsi_idx = choice == 1
    else:
        contra_idx = choice == 1
        ipsi_idx = choice == 0
    return contra_idx[-1, :], ipsi_idx[-1, :]


def recover_targ_loc(desired_out, stim_dir):
    # return the target arrangement: green_contra = 0, red_contra = 1
    choice = np.argmax(desired_out, 2)
    target = choice == 1  # contra = 0, ipsi = 1
    dir = stim_dir == 315  # green = 0, red = 1
    return np.logical_xor(target, dir)


def get_choice_color(y, desired_out, stim_dir):
    # return choice color (green = 0, red = 1)
    choice = np.argmax(y, 2)
    targ_loc = recover_targ_loc(desired_out, stim_dir)
    return np.logical_xor(choice, targ_loc)


def calc_avg_idx(h, trial_idx, cell_idx):
    return np.mean(h[:, trial_idx, cell_idx], axis=[1, 2])


def find_correct_idx(y, desired_output):
    target_max = np.argmax(desired_output, axis=2)[-1, :]
    output_max = np.argmax(y, axis=2)[-1, :]
    return target_max == output_max


def combine_idx(*args):
    temp = args[0]
    for i in range(1, len(args)):
        if args[i] is not None:
            if temp is not None:
                temp = np.logical_and(temp, args[i])
            else:
                temp = args[i]
    return temp


def find_pref_dir(stim_dir, h):
    red_idx = stim_dir == 315
    green_idx = stim_dir == 135
    red_mean = np.mean(h[stim_st_time:, red_idx, :], axis=(0, 1))
    green_mean = np.mean(h[stim_st_time:, green_idx, :], axis=(0, 1))
    pref_red = red_mean > green_mean
    return pref_red


def find_pref_sac(y, h):
    choice = np.argmax(y, 2)[-1, :]
    contra_idx = choice == 0
    ipsi_idx = choice == 1
    pref_ipsi = []
    for i in range(h.shape[2]):
        contra_mean = np.mean(h[stim_st_time:, contra_idx, i])
        ipsi_mean = np.mean(h[stim_st_time:, ipsi_idx, i])
        pref_ipsi.append(contra_mean < ipsi_mean)
    return pref_ipsi, choice


def create_grp_mask(mask_shape, grp_idx):
    mask = np.zeros(mask_shape).astype(bool)
    mask[:, grp_idx[0] : grp_idx[1]] = True
    return mask


def get_temp_h(coh_idx, h, y, pref_dir, m1_idx, m2_idx, mode, correct_idx=None):
    if mode == "dir":
        contra_idx_m1, ipsi_idx_m1 = find_sac_idx(y, True)
        contra_idx_m2, ipsi_idx_m2 = find_sac_idx(y, False)
    elif mode == "sac":
        contra_idx_m1, ipsi_idx_m1, contra_idx_m2, ipsi_idx_m2 = None, None, None, None

    m1_mask = create_grp_mask(pref_dir.shape, m1_idx)
    m2_mask = create_grp_mask(pref_dir.shape, m2_idx)

    ipsi_pref_idx_m1 = (
        pref_dir
        * np.broadcast_to(
            combine_idx(ipsi_idx_m1, coh_idx, correct_idx)[:, None], pref_dir.shape
        )
        * m1_mask
    )
    ipsi_nonpref_idx_m1 = (
        (~pref_dir)
        * np.broadcast_to(
            combine_idx(ipsi_idx_m1, coh_idx, correct_idx)[:, None], pref_dir.shape
        )
        * m1_mask
    )
    ipsi_pref_idx_m2 = (
        pref_dir
        * np.broadcast_to(
            combine_idx(ipsi_idx_m2, coh_idx, correct_idx)[:, None], pref_dir.shape
        )
        * m2_mask
    )
    ipsi_nonpref_idx_m2 = (
        (~pref_dir)
        * np.broadcast_to(
            combine_idx(ipsi_idx_m2, coh_idx, correct_idx)[:, None], pref_dir.shape
        )
        * m2_mask
    )
    if mode == "dir":
        ipsi_h_pref = np.append(h[:, ipsi_pref_idx_m1], h[:, ipsi_pref_idx_m2], axis=1)
        ipsi_h_nonpref = np.append(
            h[:, ipsi_nonpref_idx_m1], h[:, ipsi_nonpref_idx_m2], axis=1
        )
        contra_pref_idx_m1 = (
            pref_dir
            * np.broadcast_to(
                combine_idx(contra_idx_m1, coh_idx, correct_idx)[:, None],
                pref_dir.shape,
            )
            * m1_mask
        )
        contra_nonpref_idx_m1 = (
            (~pref_dir)
            * np.broadcast_to(
                combine_idx(contra_idx_m1, coh_idx, correct_idx)[:, None],
                pref_dir.shape,
            )
            * m1_mask
        )
        contra_pref_idx_m2 = (
            pref_dir
            * np.broadcast_to(
                combine_idx(contra_idx_m2, coh_idx, correct_idx)[:, None],
                pref_dir.shape,
            )
            * m2_mask
        )
        contra_nonpref_idx_m2 = (
            (~pref_dir)
            * np.broadcast_to(
                combine_idx(contra_idx_m2, coh_idx, correct_idx)[:, None],
                pref_dir.shape,
            )
            * m2_mask
        )
        contra_h_pref = np.append(
            h[:, contra_pref_idx_m1], h[:, contra_pref_idx_m2], axis=1
        )
        contra_h_nonpref = np.append(
            h[:, contra_nonpref_idx_m1], h[:, contra_nonpref_idx_m2], axis=1
        )
        return ipsi_h_pref, ipsi_h_nonpref, contra_h_pref, contra_h_nonpref
    elif mode == "sac":
        return (
            h[:, ipsi_pref_idx_m1],
            h[:, ipsi_nonpref_idx_m1],
            h[:, ipsi_pref_idx_m2],
            h[:, ipsi_nonpref_idx_m2],
        )


def plot_dir_selectivity(
    h, m1_idx, m2_idx, y, desired_out, stim_level, stim_dir, title, save_plt
):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    # find the trial of preferred direction
    pref_red = find_pref_dir(stim_dir, h)
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


def min_max_normalize(arr):
    norm_arr = (arr - np.min(np.mean(arr, axis=1), axis=0)) / (
        np.max(np.mean(arr, axis=1), axis=0) - np.min(np.mean(arr, axis=1), axis=0)
    )
    return norm_arr


def plot_sac_selectivity(
    h, m1_idx, m2_idx, y, desired_out, stim_level, title, save_plt
):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    correct_idx = find_correct_idx(y, desired_out)
    # find the trial of preferred direction
    pref_ipsi, choice = find_pref_sac(y, h)
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


for rep in all_rep:
    for lr in all_lr:
        main(lr, rep)
