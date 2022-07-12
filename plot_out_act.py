import numpy as np
import tables
import matplotlib.pyplot as plt
import os
from utils import *


f_dir = "gamma_inout_reinit_model"
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
    y = test_table["y_hist_iter%d" % max_iter][:]
    desired_out = test_table["target_iter%d" % max_iter][:]
    stim_level = test_table["stim_level_iter%d" % max_iter][:]
    stim_dir = test_table["stim_dir_iter%d" % max_iter][:]
    desired_out, stim_dir = correct_zero_coh(y, stim_level, stim_dir, desired_out)
    plot_output_sac_selectivity(y, desired_out, stim_level, 'Ouput_saccade_selectivity_rep%d_lr%f'%(rep, lr), True)


def get_temp_y(coh_idx, y, choice, correct_idx=None):

    left_idx = combine_idx(~choice, coh_idx, correct_idx)
    right_idx = combine_idx(choice, coh_idx, correct_idx)

    y_left_m1 = y[:, left_idx, 0]
    y_right_m1 = y[:, right_idx, 0]
    y_left_m2 = y[:, left_idx, 1]
    y_right_m2 = y[:, right_idx, 1]

    return y_left_m1, y_right_m1, y_left_m2, y_right_m2
    
def plot_output_sac_selectivity(y, desired_out, stim_level, title, save_plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    correct_idx = find_correct_idx(y, desired_out)
    choice = np.argmax(y, 2)[-1, :]

    # zero coherence stimulus direction is based on the choice color
    if sum(Z_idx) != 0:
        h_left_m1, h_right_m1, h_left_m2, h_right_m2 = get_temp_y(Z_idx, y, choice)
        ax1.plot(
            np.mean(h_right_m1, axis=1),
            linestyle="--",
            color="k",
            label="right, Z",
        )
        ax1.plot(np.mean(h_left_m1, axis=1), color="k", label="left, Z")
        ax2.plot(
            np.mean(h_right_m2, axis=1),
            linestyle="--",
            color="k",
            label="right, Z",
        )
        ax2.plot(np.mean(h_left_m2, axis=1), color="k", label="left, Z")

    h_left_m1, h_right_m1, h_left_m2, h_right_m2 = get_temp_y(L_idx, y, choice, correct_idx)
    ax1.plot(
        np.mean(h_right_m1, axis=1), linestyle="--", color="b", label="right, L"
    )
    ax1.plot(np.mean(h_left_m1, axis=1), color="b", label="left, L")
    ax2.plot(
        np.mean(h_right_m2, axis=1), linestyle="--", color="b", label="right, L"
    )
    ax2.plot(np.mean(h_left_m2, axis=1), color="b", label="left, L")

    h_left_m1, h_right_m1, h_left_m2, h_right_m2 = get_temp_y(M_idx, y, choice, correct_idx)
    ax1.plot(
        np.mean(h_right_m1, axis=1), linestyle="--", color="g", label="right, M"
    )
    ax1.plot(np.mean(h_left_m1, axis=1), color="g", label="left, M")
    ax2.plot(
        np.mean(h_right_m2, axis=1), linestyle="--", color="g", label="right, M"
    )
    ax2.plot(np.mean(h_left_m2, axis=1), color="g", label="left, M")

    h_left_m1, h_right_m1, h_left_m2, h_right_m2 = get_temp_y(H_idx, y, choice, correct_idx)
    ax1.plot(
        np.mean(h_right_m1, axis=1), linestyle="--", color="r", label="right, H"
    )
    ax1.plot(np.mean(h_left_m1, axis=1), color="r", label="left, H")
    ax2.plot(
        np.mean(h_right_m2, axis=1), linestyle="--", color="r", label="right, H"
    )
    ax2.plot(np.mean(h_left_m2, axis=1), color="r", label="left, H")

    ax1.set_title("Module 1")
    ax1.set_ylabel("Average activity")
    ax1.set_xlabel("Time")
    ax1.axvline(x=target_st_time, color="k")
    ax1.axvline(x=stim_st_time, color="k")

    ax2.set_title("Module 2")
    ax2.set_ylabel("Average activity")
    ax2.set_xlabel("Time")
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax2.axvline(x=target_st_time, color="k")
    ax2.axvline(x=stim_st_time, color="k")

    plt.suptitle(title)

    if save_plt:
        pic_dir = os.path.join(
            f_dir, "output_neuron_act"
        )
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.png" % (title)))
        plt.close(fig)


for rep in all_rep:
    for lr in all_lr:
        main(lr, rep)
