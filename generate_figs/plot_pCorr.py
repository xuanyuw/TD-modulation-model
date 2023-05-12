import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_ROC import get_sel_cells
from types import SimpleNamespace
from utils import (
    find_coh_idx,
    find_sac_idx,
    combine_idx,
    load_test_data,
    min_max_normalize,
    get_choice_color,
    get_pref_idx,
    find_pref_targ_color_motion_cell,
)
from scipy.stats import ttest_rel, sem
from scipy.io import savemat, loadmat
from tqdm import tqdm

# plot settings

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.figsize"] = [10, 4]

f_dir = "F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
total_rep = 1
# f_dir = "/Users/xuanyuwu/Documents/GitHub/TD-modulation-model/crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
all_rep = range(total_rep)
lr = 0.02

stim_st_time = 45
target_st_time = 25
normalize = True
plot_sel = True
rerun_calc = False
save_plot = True

# define plot window
st = 17
win = 5
h_len = 70 - 19 - 3


def main():
    m1_rng = np.concatenate((np.arange(0, 40), np.arange(160, 170)))
    m2_rng = np.concatenate((np.arange(80, 120), np.arange(180, 190)))
    motion_selective, _ = get_sel_cells()
    if rerun_calc:
        pbar = tqdm(total=total_rep)
        for rep in all_rep:
            extract_pCorr_data(rep, m1_rng, True, motion_selective[rep, m1_rng])
            extract_pCorr_data(rep, m2_rng, False, motion_selective[rep, m2_rng])
            pbar.update(1)
        return  # use matlab to calculate partial correlation
    else:
        if total_rep == 1:
            plot_pCorr_data(0)
        else:
            plot_pCorr_data(all_rep)


def plot_pCorr_data(rep):
    if isinstance(rep, int):
        pCorr_results = loadmat(
            os.path.join(f_dir, "pCorr_data", "pCorr_result_rep%d.mat" % (rep))
        )
        ipsi_pCorr_stim = pCorr_results["ipsi_pCorr_stim"]
        contra_pCorr_stim = pCorr_results["contra_pCorr_stim"]
        ipsi_pCorr_choice = pCorr_results["ipsi_pCorr_choice"]
        contra_pCorr_choice = pCorr_results["contra_pCorr_choice"]
    else:
        ipsi_pCorr_stim = np.zeros((h_len, len(rep)))
        contra_pCorr_stim = np.zeros((h_len, len(rep)))
        ipsi_pCorr_choice = np.zeros((h_len, len(rep)))
        contra_pCorr_choice = np.zeros((h_len, len(rep)))
        for r in rep:
            pCorr_results = loadmat(
                os.path.join(f_dir, "pCorr_data", "pCorr_result_rep%d.mat" % (r))
            )
            ipsi_pCorr_stim[:, r] = np.mean(pCorr_results["ipsi_pCorr_stim"], axis=1)
            contra_pCorr_stim[:, r] = np.mean(
                pCorr_results["contra_pCorr_stim"], axis=1
            )
            ipsi_pCorr_choice[:, r] = np.mean(
                pCorr_results["ipsi_pCorr_choice"], axis=1
            )
            contra_pCorr_choice[:, r] = np.mean(
                pCorr_results["contra_pCorr_choice"], axis=1
            )

    ipsi_stim_ste = sem(ipsi_pCorr_stim, axis=1)
    contra_stim_ste = sem(contra_pCorr_stim, axis=1)
    ipsi_choice_ste = sem(ipsi_pCorr_choice, axis=1)
    contra_choice_ste = sem(contra_pCorr_choice, axis=1)

    ipsi_stim_mean = np.mean(ipsi_pCorr_stim, axis=1)
    contra_stim_mean = np.mean(contra_pCorr_stim, axis=1)
    ipsi_choice_mean = np.mean(ipsi_pCorr_choice, axis=1)
    contra_choice_mean = np.mean(contra_pCorr_choice, axis=1)

    data_dict = {
        "ipsi_pCorr_stim": ipsi_pCorr_stim,
        "contra_pCorr_stim": contra_pCorr_stim,
        "ipsi_pCorr_choice": ipsi_pCorr_choice,
        "contra_pCorr_choice": contra_pCorr_choice,
        "ipsi_stim_mean": ipsi_stim_mean,
        "contra_stim_mean": contra_stim_mean,
        "ipsi_choice_mean": ipsi_choice_mean,
        "contra_choice_mean": contra_choice_mean,
        "ipsi_stim_ste": ipsi_stim_ste,
        "contra_stim_ste": contra_stim_ste,
        "ipsi_choice_ste": ipsi_choice_ste,
        "contra_choice_ste": contra_choice_ste,
    }
    create_plot(data_dict)


def create_plot(data_dict):
    ipsi_pCorr_stim = data_dict["ipsi_pCorr_stim"]
    contra_pCorr_stim = data_dict["contra_pCorr_stim"]
    ipsi_pCorr_choice = data_dict["ipsi_pCorr_choice"]
    contra_pCorr_choice = data_dict["contra_pCorr_choice"]
    ipsi_stim_mean = data_dict["ipsi_stim_mean"]
    contra_stim_mean = data_dict["contra_stim_mean"]
    ipsi_choice_mean = data_dict["ipsi_choice_mean"]
    contra_choice_mean = data_dict["contra_choice_mean"]
    ipsi_stim_ste = data_dict["ipsi_stim_ste"]
    contra_stim_ste = data_dict["contra_stim_ste"]
    ipsi_choice_ste = data_dict["ipsi_choice_ste"]
    contra_choice_ste = data_dict["contra_choice_ste"]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    xticks = np.array([0, round(h_len / 2), h_len])

    ax1.fill_between(
        np.arange(h_len),
        ipsi_choice_mean - ipsi_choice_ste,
        ipsi_choice_mean + ipsi_choice_ste,
        color="#808080",
        alpha=0.5,
    )
    ax1.fill_between(
        np.arange(h_len),
        contra_choice_mean - contra_choice_ste,
        contra_choice_mean + contra_choice_ste,
        color="#808080",
        alpha=0.5,
    )
    ax1.plot(ipsi_choice_mean, color="r", label="ipsi")
    ax1.plot(contra_choice_mean, color="b", label="contra")

    stim_pval = ttest_rel(ipsi_pCorr_stim, contra_pCorr_stim, axis=1).pvalue
    choice_pval = ttest_rel(ipsi_pCorr_choice, contra_pCorr_choice, axis=1).pvalue
    stim_pval_x = np.where(stim_pval <= 0.005)[0]
    choice_pval_x = np.where(choice_pval <= 0.005)[0]

    ax1.set_xlim(0, len(ipsi_choice_mean))
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([-500, 0, 500])
    ax1.set_ylabel("R-choice")
    ax1.set_xlabel("Time to motion onset(ms)")
    ax1.axvline(x=round(h_len / 2), color="k", alpha=0.8, linestyle="--", linewidth=1)
    ax1.axvline(
        x=round(h_len / 2) - (stim_st_time - target_st_time),
        color="k",
        alpha=0.8,
        linestyle="--",
        linewidth=1,
    )
    ax1.legend(loc="best", prop={"size": 10}, frameon=False)

    ax2.fill_between(
        np.arange(len(ipsi_stim_mean)),
        ipsi_stim_mean - ipsi_stim_ste,
        ipsi_stim_mean + ipsi_stim_ste,
        color="#808080",
        alpha=0.5,
    )
    ax2.fill_between(
        np.arange(len(contra_stim_mean)),
        contra_stim_mean - contra_stim_ste,
        contra_stim_mean + contra_stim_ste,
        color="#808080",
        alpha=0.5,
    )
    ax2.plot(ipsi_stim_mean, color="r")
    ax2.plot(contra_stim_mean, color="b")

    ax2.set_xlim(0, len(ipsi_choice_mean))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([-500, 0, 500])
    ax2.set_ylabel("R-stimulus")
    ax2.set_xlabel("Time to motion onset(ms)")
    ax2.axvline(x=round(h_len / 2), color="k", alpha=0.8, linestyle="--", linewidth=1)
    ax2.axvline(
        x=round(h_len / 2) - (stim_st_time - target_st_time),
        color="k",
        alpha=0.8,
        linestyle="--",
        linewidth=1,
    )

    yl = ax1.get_ylim()
    pval_y = max(yl) + 0.005
    ax1.scatter(
        choice_pval_x,
        np.ones(choice_pval_x.shape) * pval_y,
        color="k",
        marker="*",
        linewidths=2,
    )
    ax2.scatter(
        stim_pval_x,
        np.ones(stim_pval_x.shape) * pval_y,
        color="k",
        marker="*",
        linewidths=2,
    )

    plt.tight_layout()

    if save_plot:
        pic_dir = os.path.join(f_dir, "pCorr_plots")
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "pCorr_%dnet.png" % total_rep))
        plt.savefig(os.path.join(pic_dir, "pCorr_%dnet.pdf" % total_rep), format="pdf")
        plt.savefig(os.path.join(pic_dir, "pCorr_%dnet.eps" % total_rep), format="eps")
        plt.close(fig)


def extract_pCorr_data(
    rep,
    motion_rng,
    m1,
    m_sel,
    coh_code={"H": 4, "M": 2, "L": 1, "Z": 0},
    pref_sign=1,
    non_sign=-1,
):
    n = SimpleNamespace(
        **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
    )
    normalized_h = min_max_normalize(n.h)
    if normalize:
        h = normalized_h
    else:
        h = n.h

    coh_levels = ["H", "M", "L", "Z"]
    coh_dict = find_coh_idx(n.stim_level)
    # extract indices for each case
    contra_idx, ipsi_idx = find_sac_idx(n.y, m1)
    # red_motion_idx = n.stim_dir==315
    # green_motion_idx = n.stim_dir==135
    pref_dir, _ = get_pref_idx(n, h)
    pref_dir = pref_dir[:, motion_rng][:, m_sel.astype(bool)]

    pref_targ_c = find_pref_targ_color_motion_cell(h, n)
    pref_targ_c = pref_targ_c[:, motion_rng][:, m_sel.astype(bool)]

    # populate stim arr
    ipsi_stim_arr = np.zeros(pref_dir.shape)
    contra_stim_arr = np.zeros(pref_dir.shape)

    # cor_idx = np.tile(n.correct_idx, (pref_dir.shape[1], 1)).T
    ipsi_idx = np.tile(ipsi_idx, (pref_dir.shape[1], 1)).T
    contra_idx = np.tile(contra_idx, (pref_dir.shape[1], 1)).T

    for coh in coh_levels:
        coh_idx = np.tile(coh_dict[coh], (pref_dir.shape[1], 1)).T
        ipsi_stim_pref_idx = combine_idx(pref_dir, coh_idx, ipsi_idx)
        ipsi_stim_non_idx = combine_idx(~pref_dir, coh_idx, ipsi_idx)
        contra_stim_pref_idx = combine_idx(pref_dir, coh_idx, contra_idx)
        contra_stim_non_idx = combine_idx(~pref_dir, coh_idx, contra_idx)
        ipsi_stim_arr[ipsi_stim_pref_idx] = coh_code[coh] * pref_sign
        ipsi_stim_arr[ipsi_stim_non_idx] = coh_code[coh] * non_sign
        contra_stim_arr[contra_stim_pref_idx] = coh_code[coh] * pref_sign
        contra_stim_arr[contra_stim_non_idx] = coh_code[coh] * non_sign

    # populate choice arr
    ipsi_choice_arr = np.zeros(pref_targ_c.shape)
    contra_choice_arr = np.zeros(pref_targ_c.shape)
    ipsi_choice_arr[combine_idx(pref_targ_c, ipsi_idx)] = pref_sign * 2
    ipsi_choice_arr[combine_idx(~pref_targ_c, ipsi_idx)] = non_sign * 2
    contra_choice_arr[combine_idx(pref_targ_c, contra_idx)] = pref_sign * 2
    contra_choice_arr[combine_idx(~pref_targ_c, contra_idx)] = non_sign * 2

    # average fr to reduce noise
    h_temp = h[st:, :, motion_rng][:, :, m_sel.astype(bool)]
    h_final = np.zeros((h_temp.shape[0] - win, h_temp.shape[1], h_temp.shape[2]))
    for j in range(h_temp.shape[0] - win):
        h_final[j, :, :] = np.mean(h_temp[j : j + win, :, :], axis=0)

    # save data needed for pcorr calc
    data_dict = {
        "h": h_final,
        "ipsi_stim_arr": ipsi_stim_arr,
        "contra_stim_arr": contra_stim_arr,
        "ipsi_choice_arr": ipsi_choice_arr,
        "contra_choice_arr": contra_choice_arr,
    }

    if not os.path.exists(os.path.join(f_dir, "pCorr_data")):
        os.makedirs(os.path.join(f_dir, "pCorr_data"))

    if m1:
        savemat(
            os.path.join(f_dir, "pCorr_data", "pCorr_data_rep%d_m1.mat" % rep),
            data_dict,
        )
    else:
        savemat(
            os.path.join(f_dir, "pCorr_data", "pCorr_data_rep%d_m2.mat" % rep),
            data_dict,
        )


main()
