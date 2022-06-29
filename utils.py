import numpy as np
from calc_params import par


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


def find_pref_dir(stim_level, stim_dir, h, stim_st_time):
    nonZ_idx = np.array(stim_level) != b"Z"
    red_idx = stim_dir == 315
    green_idx = stim_dir == 135
    red_mean = np.mean(h[stim_st_time:, (red_idx & nonZ_idx), :], axis=(0, 1))
    green_mean = np.mean(h[stim_st_time:, (green_idx & nonZ_idx), :], axis=(0, 1))
    pref_red = red_mean > green_mean
    return pref_red


def find_pref_sac(y, h, stim_st_time):
    choice = np.argmax(y, 2)[-1, :]
    contra_idx = choice == 0
    ipsi_idx = choice == 1
    pref_ipsi = []
    for i in range(h.shape[2]):
        contra_mean = np.mean(h[stim_st_time:, contra_idx, i])
        ipsi_mean = np.mean(h[stim_st_time:, ipsi_idx, i])
        pref_ipsi.append(contra_mean < ipsi_mean)
    return pref_ipsi, choice


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
