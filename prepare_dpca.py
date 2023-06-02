import os
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from utils import load_test_data, min_max_normalize, recover_targ_loc
from scipy.io import savemat
from utils import combine_idx

lr = 0.02
total_rep = 50
correct_only = False

# f_dirs = [
#     "test_output_full_model",
#     "test_output_noFeedback_model",
#     "cutSpec_model",
#     "cutNonspec_model",
# ]

f_dirs = ["trained_eqNum_removeFB_model"]

data_dir = "dPCA_allTrial_data"


def main():
    for f_dir in f_dirs:
        model_type = f_dir.split("_")[-2]
        if not os.path.exists(os.path.join(data_dir, model_type, "allTrials")):
            os.makedirs(os.path.join(data_dir, model_type, "allTrials"))
        # if not os.path.exists(os.path.join(data_dir, model_type, "sepStim")):
        #     os.makedirs(os.path.join(data_dir, model_type, "sepStim"))
        # if not os.path.exists(os.path.join(data_dir, model_type, "sepSac")):
        #     os.makedirs(os.path.join(data_dir, model_type, "sepSac"))
        for rep in tqdm(range(total_rep)):
            save_fn = os.path.join(data_dir, model_type, "allTrials", "rep%d.mat" % rep)
            # if correct_only:
            #     save_fn_sepStim = os.path.join(
            #         data_dir, model_type, "sepStim", "rep%d_correct.mat" % rep
            #     )
            #     save_fn_sepSac = os.path.join(
            #         data_dir, model_type, "sepSac", "rep%d_correct.mat" % rep
            #     )
            # else:
            #     save_fn_sepStim = os.path.join(
            #         data_dir, model_type, "sepStim", "rep%d.mat" % rep
            #     )
            #     save_fn_sepSac = os.path.join(
            #         data_dir, model_type, "sepSac", "rep%d.mat" % rep
            #     )
            n = SimpleNamespace(
                **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
            )
            reshaped_data = reshape_data(n)
            # reshaped_data_sepStim, reshaped_data_sepSac = reshape_data(n)
            savemat(save_fn, {"data": reshaped_data})
            # savemat(save_fn_sepSac, {"data": reshaped_data_sepSac})


def reshape_data(n):
    # reshaped_data:
    # dim 1: number of neurons
    # dim 2: number of target locations
    # dim 3: number of stimulus directions
    # dim 4: number of time steps
    # dim 5: number of trials
    normalized_h = min_max_normalize(n.h)
    normalized_h = normalized_h.transpose(2, 0, 1)
    stim_dir = n.stim_dir
    targ_arrange = recover_targ_loc(n.desired_out, n.stim_dir)[-1, :]
    # dim1 = number of neurons; dim2 = saccade direction; dim3 = stimulus direction; dim4 = target arrangement; dim5 = time; dim6 = trials
    cond1_idx = combine_idx(n.choice == 0, stim_dir == 135, targ_arrange == 0)
    cond2_idx = combine_idx(n.choice == 0, stim_dir == 135, targ_arrange == 1)
    cond3_idx = combine_idx(n.choice == 0, stim_dir == 315, targ_arrange == 0)
    cond4_idx = combine_idx(n.choice == 0, stim_dir == 315, targ_arrange == 1)

    cond5_idx = combine_idx(n.choice == 1, stim_dir == 135, targ_arrange == 0)
    cond6_idx = combine_idx(n.choice == 1, stim_dir == 135, targ_arrange == 1)
    cond7_idx = combine_idx(n.choice == 1, stim_dir == 315, targ_arrange == 0)
    cond8_idx = combine_idx(n.choice == 1, stim_dir == 315, targ_arrange == 1)

    max_num_trial = np.max(
        [
            np.sum(cond1_idx),
            np.sum(cond2_idx),
            np.sum(cond3_idx),
            np.sum(cond4_idx),
            np.sum(cond5_idx),
            np.sum(cond6_idx),
            np.sum(cond7_idx),
            np.sum(cond8_idx),
        ]
    )
    reshaped_data = np.empty(
        (
            normalized_h.shape[0],
            len(np.unique(n.choice)),
            len(np.unique(stim_dir)),
            len(np.unique(targ_arrange)),
            normalized_h.shape[1],
            max_num_trial,
        )
    )
    reshaped_data[:] = np.nan

    reshaped_data[:, 0, 0, 0, :, : np.sum(cond1_idx)] = normalized_h[:, :, cond1_idx]
    reshaped_data[:, 0, 0, 1, :, : np.sum(cond2_idx)] = normalized_h[:, :, cond2_idx]
    reshaped_data[:, 0, 1, 0, :, : np.sum(cond3_idx)] = normalized_h[:, :, cond3_idx]
    reshaped_data[:, 0, 1, 1, :, : np.sum(cond4_idx)] = normalized_h[:, :, cond4_idx]
    reshaped_data[:, 1, 0, 0, :, : np.sum(cond5_idx)] = normalized_h[:, :, cond5_idx]
    reshaped_data[:, 1, 0, 1, :, : np.sum(cond6_idx)] = normalized_h[:, :, cond6_idx]
    reshaped_data[:, 1, 1, 0, :, : np.sum(cond7_idx)] = normalized_h[:, :, cond7_idx]
    reshaped_data[:, 1, 1, 1, :, : np.sum(cond8_idx)] = normalized_h[:, :, cond8_idx]

    return reshaped_data

    # if correct_only:
    #     cond1_idx = combine_idx(stim_dir == 135, targ_arrange == 0, n.correct_idx)

    #     cond2_idx = combine_idx(stim_dir == 135, targ_arrange == 1, n.correct_idx)

    #     cond3_idx = combine_idx(stim_dir == 315, targ_arrange == 0, n.correct_idx)

    #     cond4_idx = combine_idx(stim_dir == 315, targ_arrange == 1, n.correct_idx)

    # cond5_idx = combine_idx(n.choice == 0, targ_arrange == 0, n.correct_idx)

    # cond6_idx = combine_idx(n.choice == 0, targ_arrange == 1, n.correct_idx)

    # cond7_idx = combine_idx(n.choice == 1, targ_arrange == 0, n.correct_idx)

    # cond8_idx = combine_idx(n.choice == 1, targ_arrange == 1, n.correct_idx)
    # else:
    #     cond1_idx = combine_idx(stim_dir == 135, targ_arrange == 0)

    #     cond2_idx = combine_idx(stim_dir == 135, targ_arrange == 1)

    #     cond3_idx = combine_idx(stim_dir == 315, targ_arrange == 0)

    #     cond4_idx = combine_idx(stim_dir == 315, targ_arrange == 1)

    #     cond5_idx = combine_idx(n.choice == 0, targ_arrange == 0)

    #     cond6_idx = combine_idx(n.choice == 0, targ_arrange == 1)

    #     cond7_idx = combine_idx(n.choice == 1, targ_arrange == 0)

    #     cond8_idx = combine_idx(n.choice == 1, targ_arrange == 1)

    # max_num_trial1 = np.max(
    #     [
    #         np.sum(cond1_idx),
    #         np.sum(cond2_idx),
    #         np.sum(cond3_idx),
    #         np.sum(cond4_idx),
    #     ]
    # )
    # max_num_trial2 = np.max(
    #     [np.sum(cond5_idx), np.sum(cond6_idx), np.sum(cond7_idx), np.sum(cond8_idx)]
    # )
    # min_num_trial1 = np.min(
    #     [
    #         np.sum(cond1_idx),
    #         np.sum(cond2_idx),
    #         np.sum(cond3_idx),
    #         np.sum(cond4_idx),
    #     ]
    # )
    # min_num_trial2 = np.min(
    #     [np.sum(cond5_idx), np.sum(cond6_idx), np.sum(cond7_idx), np.sum(cond8_idx)]
    # )
    # assert min_num_trial1 > 0 and min_num_trial2 > 0
    # reshaped_data_sepStim = np.empty(
    #     (
    #         normalized_h.shape[0],
    #         len(np.unique(stim_dir)),
    #         len(np.unique(targ_arrange)),
    #         normalized_h.shape[1],
    #         max_num_trial1,
    #     )
    # )
    # reshaped_data_sepStim[:] = np.nan
    # reshaped_data_sepStim[:, 0, 0, :, : np.sum(cond1_idx)] = normalized_h[
    #     :, :, cond1_idx
    # ]
    # reshaped_data_sepStim[:, 0, 1, :, : np.sum(cond2_idx)] = normalized_h[
    #     :, :, cond2_idx
    # ]
    # reshaped_data_sepStim[:, 1, 0, :, : np.sum(cond3_idx)] = normalized_h[
    #     :, :, cond3_idx
    # ]
    # reshaped_data_sepStim[:, 1, 1, :, : np.sum(cond4_idx)] = normalized_h[
    #     :, :, cond4_idx
    # ]

    # reshaped_data_sepSac = np.empty(
    #     (
    #         normalized_h.shape[0],
    #         len(np.unique(n.choice)),
    #         len(np.unique(targ_arrange)),
    #         normalized_h.shape[1],
    #         max_num_trial2,
    #     )
    # )
    # reshaped_data_sepSac[:] = np.nan

    # reshaped_data_sepSac[:, 0, 0, :, : np.sum(cond5_idx)] = normalized_h[
    #     :, :, cond5_idx
    # ]
    # reshaped_data_sepSac[:, 0, 1, :, : np.sum(cond6_idx)] = normalized_h[
    #     :, :, cond6_idx
    # ]
    # reshaped_data_sepSac[:, 1, 0, :, : np.sum(cond7_idx)] = normalized_h[
    #     :, :, cond7_idx
    # ]
    # reshaped_data_sepSac[:, 1, 1, :, : np.sum(cond8_idx)] = normalized_h[
    #     :, :, cond8_idx
    # ]

    # return reshaped_data_sepStim, reshaped_data_sepSac
    # return reshaped_data_sepStim


if __name__ == "__main__":
    main()
