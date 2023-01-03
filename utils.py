import numpy as np
import matplotlib.pyplot as plt
from calc_params import par
import os
import tables
from scipy.stats import f_oneway
from math import ceil


STIM_ST_TIME = (par['time_fixation'] + par['time_target'])//par['dt']
TARG_ST_TIME = par['time_fixation']//par['dt']
DT = par['dt']


def find_coh_idx(stim_level):
    coh_dict = {}
    for i in np.unique(stim_level):
        coh_dict[i.decode("utf8")] = np.array(stim_level) == i
    return coh_dict


def find_sac_idx(y, m1):
    choice = np.argmax(y, 2)
    if m1:
        contra_idx = choice == 0
        ipsi_idx = choice == 1
    else:
        contra_idx = choice == 1
        ipsi_idx = choice == 0
    return contra_idx[-1, :], ipsi_idx[-1, :]


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


def relu(input):
    return input * (input > 0)


def correct_zero_coh(y, stim_level, stim_dir, desired_out):
    # change the stimulus direction and the desired output based on the choice in zero coherence trials
    zero_idx = np.array(stim_level) == b"Z"
    # convert y to the same form as the desired output
    temp_y = np.zeros(y.shape)
    temp_y[
        np.arange(y.shape[0])[:, None], np.arange(y.shape[1]), np.argmax(y, axis=2)
    ] = 1
    choice_color = get_choice_color(y, desired_out, stim_dir)[
        -1, :
    ]  # return choice color (green = 0, red = 1)
    stim_dir[zero_idx & (choice_color == 0)] = 135
    stim_dir[zero_idx & (choice_color == 1)] = 315
    # the desired output of zero coherence trials is the choice (no correct or incorrect in zero coherence case)
    desired_out[:, zero_idx, :] = temp_y[:, zero_idx, :]
    return desired_out, stim_dir


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


def find_pref_dir(stim_level, stim_dir, h, stim_st_time=STIM_ST_TIME):
    nonZ_idx = np.array(stim_level) != b"Z"
    red_idx = stim_dir == 315
    green_idx = stim_dir == 135
    red_mean = np.mean(h[stim_st_time:, (red_idx & nonZ_idx), :], axis=(0, 1))
    green_mean = np.mean(h[stim_st_time:, (green_idx & nonZ_idx), :], axis=(0, 1))
    pref_red = red_mean > green_mean
    return pref_red

def find_pref_targ_color(h, desired_out, stim_dir, m1_targ_rng, m2_targ_rng):
    # prefer green target = 0, prefer red target = 1
    targ_loc = recover_targ_loc(desired_out, stim_dir)[-1, :]
    contra_green_mean = np.mean(h[TARG_ST_TIME:STIM_ST_TIME, targ_loc==0, :], axis=(0, 1))
    contra_red_mean = np.mean(h[TARG_ST_TIME:STIM_ST_TIME, targ_loc==1, :], axis=(0, 1))
    pref_targ_color = np.zeros((h.shape[2],))
    pref_targ_color[m1_targ_rng] = contra_green_mean[m1_targ_rng] < contra_red_mean[m1_targ_rng] 
    pref_targ_color[m2_targ_rng] = contra_green_mean[m2_targ_rng] > contra_red_mean[m2_targ_rng] #ipsi-lateral targets are the opposite of contra lateral targets
    return pref_targ_color

def find_pref_targ_color_motion_cell(h, n):
    choice_c = get_choice_color(n.y, n.desired_out, n.stim_dir)[-1, :] # return choice color (green = 0, red = 1)
    pref_red = find_pref_dir(n.stim_level, n.stim_dir, h)
    choice_c_temp = np.tile(choice_c, (len(pref_red), 1)).T
    pref_red_temp = np.tile(pref_red, (len(choice_c), 1))
    return pref_red_temp == choice_c_temp


def find_pref_sac(y, h, stim_st_time=STIM_ST_TIME):
    choice = np.argmax(y, 2)[-1, :]
    contra_idx = choice == 0
    ipsi_idx = choice == 1
    pref_ipsi = []
    for i in range(h.shape[2]):
        contra_mean = np.mean(h[stim_st_time:, contra_idx, i])
        ipsi_mean = np.mean(h[stim_st_time:, ipsi_idx, i])
        pref_ipsi.append(contra_mean < ipsi_mean)
    return pref_ipsi, choice



def get_pref_idx(n, h):
    # find the trial of preferred saccade direction
    pref_ipsi, choice = find_pref_sac(n.y, h)
    pref_ipsi_temp = np.tile(pref_ipsi, (len(choice), 1))
    choice_temp = np.tile(np.reshape(choice, (-1, 1)), (1, len(pref_ipsi)))
    pref_sac = choice_temp == pref_ipsi_temp

    # find the trial of preferred motion direction
    pref_red = find_pref_dir(n.stim_level, n.stim_dir, h)
    dir_red = n.stim_dir == 315
    pref_red_temp = np.tile(pref_red, (len(dir_red), 1))
    dir_red_temp = np.tile(np.reshape(dir_red, (-1, 1)), (1, len(pref_red)))
    pref_dir = pref_red_temp == dir_red_temp
    return pref_dir, pref_sac



def min_max_normalize(arr):
    norm_arr = (arr - np.min(np.mean(arr, axis=1), axis=0)) / (
        np.max(np.mean(arr, axis=1), axis=0) - np.min(np.mean(arr, axis=1), axis=0)
    )
    return norm_arr


def get_module_idx():
    exc_bnds = np.cumsum(
        np.insert(
            ((par["n_hidden"] * par["exc_inh_prop"]) * np.array(par["RF_perc"])).astype(
                "int"
            ),
            0,
            0,
        )
    )
    exc_idx = [(exc_bnds[i], exc_bnds[i + 1]) for i in range(len(exc_bnds) - 1)]
    inh_bnds = np.around(
        np.cumsum(
            np.insert(
                (
                    (par["n_hidden"] * (1 - par["exc_inh_prop"]))
                    * np.array(par["RF_perc"])
                ),
                0,
                0,
            )
        )
    ).astype("int")
    inh_bnds = np.max(exc_bnds) + inh_bnds
    inh_idx = [(inh_bnds[i], inh_bnds[i + 1]) for i in range(len(inh_bnds) - 1)]
    exc_idx.extend(inh_idx)
    return exc_idx


def create_grp_mask(mask_shape, grp_idx, selectivity=None):
    mask = np.zeros(mask_shape).astype(bool)
    if len(grp_idx) != 1:
        idx = []
        for g_idx in grp_idx:
            if selectivity is not None:
                idx = np.append(idx, np.intersect1d(
                    np.arange(g_idx[0], g_idx[1]), np.where(selectivity)[0]
                ))
            else:
                idx = np.append(idx, np.arange(g_idx[0], g_idx[1]))
    else:
        if selectivity is not None:
            idx = np.intersect1d(
                np.arange(grp_idx[0], grp_idx[1]), np.where(selectivity)[0]
            )
        else:
            idx = np.arange(grp_idx[0], grp_idx[1])
    idx =  idx.astype(int)
    mask[:, idx] = True
    return mask


def get_temp_h_avg(
    coh_idx, h, y, pref_dir, m1_idx, m2_idx, mode, correct_idx=None, selectivity=None
):
    assert type(mode) is str
    if mode == "dir":
        contra_idx_m1, ipsi_idx_m1 = find_sac_idx(y, True)
        contra_idx_m2, ipsi_idx_m2 = find_sac_idx(y, False)
    elif mode == "sac":
        contra_idx_m1, ipsi_idx_m1, contra_idx_m2, ipsi_idx_m2 = None, None, None, None

    m1_mask = create_grp_mask(pref_dir.shape, m1_idx, selectivity)
    m2_mask = create_grp_mask(pref_dir.shape, m2_idx, selectivity)

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
        # ipsi_h_pref = h[:, ipsi_pref_idx_m1]
        # ipsi_h_nonpref = h[:, ipsi_nonpref_idx_m1]
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
        # contra_h_pref = h[:, contra_pref_idx_m1]
        # contra_h_nonpref = h[:, contra_nonpref_idx_m1]
        contra_h_pref = np.append(
            h[:, contra_pref_idx_m1], h[:, contra_pref_idx_m2], axis=1
        )
        contra_h_nonpref = np.append(
            h[:, contra_nonpref_idx_m1], h[:, contra_nonpref_idx_m2], axis=1
        )
        return np.mean(ipsi_h_pref, axis=1), np.mean(ipsi_h_nonpref, axis=1), np.mean(contra_h_pref, axis=1), np.mean(contra_h_nonpref, axis=1)
    elif mode == "sac":
        return (
            np.mean(h[:, ipsi_pref_idx_m1], axis=1),
            np.mean(h[:, ipsi_nonpref_idx_m1], axis=1),
            np.mean(h[:, ipsi_pref_idx_m2], axis=1),
            np.mean(h[:, ipsi_nonpref_idx_m2], axis=1),
        )


def get_diff_stim(trial_info):
    idx = np.where(trial_info["stim_dir"] == 135)[0][0]
    g_motion = trial_info["neural_input"][:, idx, :]
    idx = np.where(trial_info["stim_dir"] == 315)[0][0]
    r_motion = trial_info["neural_input"][:, idx, :]
    idx = np.where(trial_info["targ_loc"] == 0)[0][0]
    m1_g = trial_info["neural_input"][:, idx, :]
    idx = np.where(trial_info["targ_loc"] == 1)[0][0]
    m1_r = trial_info["neural_input"][:, idx, :]
    return g_motion, r_motion, m1_g, m1_r


def calc_input_sum(in_weight, in_mask, stim, module_idx):
    in_val = stim @ (in_weight * in_mask)
    m1_idx = np.hstack(
        (
            np.arange(module_idx[0][0], module_idx[0][1]),
            np.arange(module_idx[2][0], module_idx[2][1]),
        )
    )
    m2_idx = np.hstack(
        (
            np.arange(module_idx[1][0], module_idx[1][1]),
            np.arange(module_idx[3][0], module_idx[3][1]),
        )
    )
    return (np.sum(in_val[:, m1_idx]), np.sum(in_val[:, m2_idx]))


def load_test_data(f_dir, f_name):
    test_output = tables.open_file(
        os.path.join(f_dir, f_name), mode="r"
    )
    test_table = test_output.root
    max_iter = get_max_iter(test_table)
    h = test_table["h_iter%d" % max_iter][:]
    y = test_table["y_hist_iter%d" % max_iter][:]
    neural_input = test_table["neural_in_iter%d" % max_iter][:]
    choice = np.argmax(y, 2)[-1, :]
    desired_out = test_table["target_iter%d" % max_iter][:]
    stim_level = test_table["stim_level_iter%d" % max_iter][:]
    stim_dir = test_table["stim_dir_iter%d" % max_iter][:]
    desired_out, stim_dir = correct_zero_coh(y, stim_level, stim_dir, desired_out)
    correct_idx = find_correct_idx(y, desired_out)

    return {
        "h": h,
        "y": y,
        "neural_input": neural_input,
        "choice": choice,
        "desired_out": desired_out,
        "stim_level": stim_level,
        "stim_dir": stim_dir,
        "desired_out": desired_out,
        "correct_idx": correct_idx,
    }


def pick_selective_neurons(h, labels, window_st=45, window_ed=70, alpha=0.01):
    lbs = np.unique(labels)
    grp1_idx = np.where(labels == lbs[0])[0]
    grp2_idx = np.where(labels == lbs[1])[0]
    if len(grp1_idx) != len(grp2_idx):
        idx_len = min(len(grp1_idx), len(grp2_idx))
        grp1_idx = grp1_idx[:idx_len]
        grp2_idx = grp2_idx[:idx_len]
    grp1 = np.mean(h[window_st:window_ed, grp1_idx, :], axis=0)
    grp2 = np.mean(h[window_st:window_ed, grp2_idx, :], axis=0)
    _, p_vals = f_oneway(grp1, grp2)
    return p_vals < alpha


def plot_coh_popu_act(
    line_dict,
    label_dict,
    coh_levels,
    color_dict={"Z": "k", "L": "b", "M": "g", "H": "r"},
    target_st_time=TARG_ST_TIME,
    stim_st_time=STIM_ST_TIME,
):

    assert all(
        k in ["dash", "solid", "ax1_title", "ax2_title", "sup_title"]
        for k in label_dict.keys()
    )
    assert len(line_dict) % 2 == 0 and len(line_dict) >= 2
    assert all("_dash" in k or "_solid" in k for k in line_dict.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for i in range(len(coh_levels)):
        if coh_levels[i] + "_dash_ax1" not in line_dict.keys():
            continue
        ax1.plot(
            line_dict[coh_levels[i] + "_dash_ax1"],
            linestyle="--",
            color=color_dict[coh_levels[i]],
            label=coh_levels[i] + "," + label_dict["dash"],
        )
        ax2.plot(
            line_dict[coh_levels[i] + "_dash_ax2"],
            linestyle="--",
            color=color_dict[coh_levels[i]],
            label=coh_levels[i] + "," + label_dict["dash"],
        )
        ax1.plot(
            line_dict[coh_levels[i] + "_solid_ax1"],
            color=color_dict[coh_levels[i]],
            label=coh_levels[i] + "," + label_dict["solid"],
        )
        ax2.plot(
            line_dict[coh_levels[i] + "_solid_ax2"],
            color=color_dict[coh_levels[i]],
            label=coh_levels[i] + "," + label_dict["solid"],
        )
    ax1.set_title(label_dict["ax1_title"])
    ax1.set_ylabel("Average activity")
    ax1.set_xlabel("Time")
    xticks = np.array([0, 25, 50])
    ax1.set_xlim(0, 50)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels((xticks+20-stim_st_time)*DT)
    ax1.axvline(x=target_st_time-20, color="k", linewidth=1, linestyle="--")
    ax1.axvline(x=stim_st_time-20, color="k", linewidth=1, linestyle="--")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.set_title(label_dict["ax2_title"])
    ax2.set_xlabel("Time")
    ax2.set_xlim(0, 50)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels((xticks+20-stim_st_time)*DT)
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={'size': 10}, frameon=False)
    ax2.axvline(x=target_st_time-20, color="k", linewidth=1, linestyle="--")
    ax2.axvline(x=stim_st_time-20, color="k", linewidth=1, linestyle="--")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle(label_dict["sup_title"])

    plt.tight_layout()
    return fig

def get_sac_avg_h(
    coh_idx, h, choice, m1_idx, m2_idx, correct_idx=None, selectivity=None
):
    choice = choice.astype(bool)
    left_idx = np.where(combine_idx(~choice, coh_idx, correct_idx))[0]
    right_idx = np.where(combine_idx(choice, coh_idx, correct_idx))[0]

    if len(m1_idx) != 1:
        m1_cells = []
        m2_cells = []
        for i in range(len(m1_idx)):
            if selectivity is not None:
                m1_cells = np.append(m1_cells, np.intersect1d(
                    np.arange(m1_idx[i][0], m1_idx[i][1]), np.where(selectivity)[0]
                ))
                m2_cells = np.append(m2_cells, np.intersect1d(
                    np.arange(m2_idx[i][0], m2_idx[i][1]), np.where(selectivity)[0]
                ))
            else:
                m1_cells = np.append(m1_cells, np.arange(m1_idx[i][0], m1_idx[i][1]))
                m2_cells = np.append(m2_cells, np.arange(m2_idx[i][0], m2_idx[i][1]))
    else:
        if selectivity is not None:
            m1_cells = np.intersect1d(
                np.arange(m1_idx[0], m1_idx[1]), np.where(selectivity)[0]
            )
            m2_cells = np.intersect1d(
                np.arange(m2_idx[0], m2_idx[1]), np.where(selectivity)[0]
            )
        else:
            m1_cells = np.arange(m1_idx[0], m1_idx[1])
            m2_cells = np.arange(m2_idx[0], m2_idx[1])

    m1_cells = m1_cells.astype(int)
    m2_cells = m2_cells.astype(int)
    h_left_m1 = np.mean(h[:, left_idx, :][:, :, m1_cells], axis=(1, 2))
    h_right_m1 = np.mean(h[:, right_idx, :][:, :, m1_cells], axis=(1, 2))
    h_left_m2 = np.mean(h[:, left_idx, :][:, :, m2_cells], axis=(1, 2))
    h_right_m2 = np.mean(h[:, right_idx, :][:, :, m2_cells], axis=(1, 2))
    return h_left_m1, h_right_m1, h_left_m2, h_right_m2

def calculate_rf_rngs():
    """Generates the bounds for rf blocks"""
    ei = [par["exc_inh_prop"], 1 - par["exc_inh_prop"]]
    rf_bnd = np.append(
        0,
        np.cumsum(
            [ceil(par["n_hidden"] * eix * p) for eix in ei for p in par["RF_perc"]]
        ),
    )
    rf_rngs = [(rf_bnd[n], rf_bnd[n + 1]) for n in range(len(rf_bnd) - 1)]
    return rf_rngs

def cut_conn(conn, mask):
    rf_rngs = calculate_rf_rngs()
    for i in range(len(rf_rngs)):
        from_rng = rf_rngs[i]
        for j in range(len(rf_rngs)):
            to_rng = rf_rngs[j]
            if conn[i, j]==0:
                sz = (from_rng[1] - from_rng[0], to_rng[1] - to_rng[0])
                mask[from_rng[0] : from_rng[1], to_rng[0] : to_rng[1]] = np.zeros(sz)
    return mask

def shuffle_conn(shuffle, weight):
    rf_rngs = calculate_rf_rngs()
    for i in range(len(rf_rngs)):
        from_rng = rf_rngs[i]
        for j in range(len(rf_rngs)):
            to_rng = rf_rngs[j]
            if shuffle[i, j]==1:
                np.random.shuffle(weight[from_rng[0] : from_rng[1], to_rng[0] : to_rng[1]])
    return weight