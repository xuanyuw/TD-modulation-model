from itertools import repeat
import numpy as np
import tables
import matplotlib.pyplot as plt
import os
import pickle
import json
from stimulus import Stimulus
from calc_params import par
from utils import get_module_idx

f_dir = "test_weights_model"
all_rep = range(1)
all_lr = [0.02]

stim_st_time = 45
target_st_time = 25


def main(rep, lr):
    # test_output = tables.open_file(
    #     os.path.join(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep)), mode="r"
    # )
    # test_table = test_output.root

    with open(os.path.join(f_dir, "init_weight_%d_lr%f.pth" % (rep, lr)), "rb") as f:
        all_weights = np.load(f, allow_pickle=True)
    all_weights = all_weights.item()
    stim = Stimulus(par)
    trial_info = stim.generate_trial()
    input_stats = check_in_weight(
        all_weights["w_in0"], all_weights["in_mask_init"], trial_info
    )
    output_stats = check_out_weight(
        all_weights["w_out0"], all_weights["out_mask_init"]
    )
    with open(
        os.path.join(f_dir, "input_output_weight_stats_rep%d.json" % rep),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump([input_stats, output_stats], f)


def check_out_weight(out_weight, out_mask):
    masked_weight = out_weight * out_mask
    masked_weight[masked_weight == 0] = np.NaN
    c0_mean = round(np.nanmean(masked_weight[:, 0]).astype("float"), 3)
    # c0_std = round(np.nanstd(masked_weight[:, 0]).astype("float"), 3)
    c1_mean = round(np.nanmean(masked_weight[:, 1]).astype("float"), 3)
    # c1_std = round(np.nanstd(masked_weight[:, 1]).astype("float"), 3)
    c0_sum = round(np.nansum(masked_weight[:, 0]).astype("float"), 3)
    c1_sum = round(np.nansum(masked_weight[:, 1]).astype("float"), 3)
    out = {
        "c0": {'# conns': int(np.sum(out_mask[:, 0])), 'sum': c0_sum, 'mean': c0_mean},  # [c0_mean, c0_std],
        "c1": {'# conns': int(np.sum(out_mask[:, 1])), 'sum': c1_sum, 'mean': c1_mean} # [c1_mean, c1_std],
    }
    return out


def check_in_weight(in_weight, in_mask, trial_info):
    g_motion, r_motion, m1_g, m1_r = get_diff_stim(trial_info)
    all_module_idx = get_module_idx()
    motion_rf_idx = [0, 2, 4, 6]
    g_motion_vals = calc_input_sum(
        in_weight, in_mask, g_motion, [all_module_idx[x] for x in motion_rf_idx]
    )
    r_motion_vals = calc_input_sum(
        in_weight, in_mask, r_motion, [all_module_idx[x] for x in motion_rf_idx]
    )
    m1_g_vals = calc_input_sum(
        in_weight,
        in_mask,
        m1_g,
        [
            all_module_idx[x]
            for x in range(len(all_module_idx))
            if x not in motion_rf_idx
        ],
    )
    m1_r_vals = calc_input_sum(
        in_weight,
        in_mask,
        m1_r,
        [
            all_module_idx[x]
            for x in range(len(all_module_idx))
            if x not in motion_rf_idx
        ],
    )
    out = {
        "g_motion": g_motion_vals,
        "r_motion": r_motion_vals,
        "m1_g": m1_g_vals,
        "m1_r": m1_r_vals,
    }
    return out


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
    in_val[in_val == 0] = np.NaN
    out = []
    for tup in module_idx:
        out.append(
            {
                '# conns': int(np.sum(in_mask[:, tup[0]:tup[1]])),
                'sum': round(np.nansum(in_val[:, tup[0] : tup[1]]).astype("float"), 3),
                'mean': round(np.nanmean(in_val[:, tup[0] : tup[1]]).astype("float"), 3),
                # round(np.nanstd(in_val[:, tup[0] : tup[1]]).astype("float"), 3),
            }
        )
    return out


for rep in all_rep:
    main(rep, 0.02)
