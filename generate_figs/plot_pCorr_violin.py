import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from statannotations.Annotator import Annotator
from scipy.io import loadmat
from scipy.stats import ttest_ind, ttest_rel
from plot_ROC import get_sel_cells
from tqdm import tqdm
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
from scipy.io import savemat, loadmat

# plot settings
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2


f_dir = "F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
plt_dir = os.path.join("generate_figs", "Fig6", "pCorr_plots")

total_rep = 50
total_shuf = 100
# f_dir = "/Users/xuanyuwu/Documents/GitHub/TD-modulation-model/crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
all_rep = range(total_rep)
lr = 0.02

stim_st_time = 45
target_st_time = 25
normalize = True

rerun_calc = False
reload_data = False
save_plot = False

# define plot window
st = 29
ed = 48


def main():
    if rerun_calc:
        m1_rng = np.concatenate((np.arange(0, 40), np.arange(160, 170)))
        m2_rng = np.concatenate((np.arange(80, 120), np.arange(180, 190)))
        motion_selective, _ = get_sel_cells()
        if "shuf" not in f_dir:
            pbar = tqdm(total=total_rep)
            for rep in all_rep:
                extract_pCorr_data(rep, m1_rng, True, motion_selective[rep, m1_rng])
                extract_pCorr_data(rep, m2_rng, False, motion_selective[rep, m2_rng])
                pbar.update(1)
            pbar.close()
        else:
            pbar = tqdm(total=total_rep * total_shuf)
            for rep in all_rep:
                for shuf in range(total_shuf):
                    extract_pCorr_data(
                        rep, m1_rng, True, motion_selective[rep, m1_rng], shuf
                    )
                    extract_pCorr_data(
                        rep, m2_rng, False, motion_selective[rep, m2_rng], shuf
                    )
                    pbar.update(1)
            pbar.close()
        return  # use matlab to calculate partial correlation

    if reload_data:
        ipsi_pCorr_stim = np.zeros((total_rep,))
        contra_pCorr_stim = np.zeros((total_rep,))
        ipsi_pCorr_choice = np.zeros((total_rep,))
        contra_pCorr_choice = np.zeros((total_rep,))
        for r in all_rep:
            pCorr_results = loadmat(
                os.path.join(f_dir, "pCorr_data", "single_pCorr_result_rep%d.mat" % (r))
            )
            ipsi_pCorr_stim[r] = np.mean(pCorr_results["ipsi_pCorr_stim"])
            contra_pCorr_stim[r] = np.mean(pCorr_results["contra_pCorr_stim"])
            ipsi_pCorr_choice[r] = np.mean(pCorr_results["ipsi_pCorr_choice"])
            contra_pCorr_choice[r] = np.mean(pCorr_results["contra_pCorr_choice"])
        rep = list(range(50)) * 4
        r_type = sum([["stim"] * 50 * 2, ["choice"] * 50 * 2], [])
        sac_dir = sum([["ipsi"] * 50, ["contra"] * 50], []) * 2
        mean_pCorr = np.concatenate(
            (ipsi_pCorr_stim, contra_pCorr_stim, ipsi_pCorr_choice, contra_pCorr_choice)
        )

        df = pd.DataFrame(
            {"rep": rep, "r_type": r_type, "sac_dir": sac_dir, "mean_pCorr": mean_pCorr}
        )
        df.to_csv(os.path.join(f_dir, "pCorr_comp_50net.csv"))
    else:
        df = pd.read_csv(os.path.join(f_dir, "pCorr_comp_50net.csv"))

    # calc p-value of difference
    ipsi_stim = df.loc[:49]["mean_pCorr"].to_numpy()
    contra_stim = df.loc[50:99]["mean_pCorr"].to_numpy()
    ipsi_choice = df.loc[100:149]["mean_pCorr"].to_numpy()
    contra_choice = df.loc[150:199]["mean_pCorr"].to_numpy()
    stim_diff = contra_stim - ipsi_stim
    choice_diff = contra_choice - ipsi_choice

    fig, ax = plt.subplots()
    # color_palette = {'H': '#FF0000', 'M': '#00FF00', 'L':'#0000FF', 'Z': 'k'}
    colors = ["#FF0000", "#0000FF"]
    handles = []
    sns.violinplot(
        x="r_type",
        y="mean_pCorr",
        hue="sac_dir",
        data=df,
        inner="points",
        ax=ax,
        palette=[".2", ".5"],
        hue_order=["ipsi", "contra"],
    )

    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 1:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor="black"))

    ax.legend(
        handles=[tuple(handles[1::2]), tuple(handles[::2])],
        labels=df["sac_dir"].astype("category").cat.categories.to_list(),
        handlelength=2,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="lower left",
        frameon=False,
    )

    # add statistical test results
    pairs = [
        (("stim", "ipsi"), ("stim", "contra")),
        (("choice", "ipsi"), ("choice", "contra")),
    ]

    f = open(os.path.join(plt_dir, "stat_test.txt"), "w")
    sys.stdout = f

    annot = Annotator(
        ax,
        pairs,
        data=df,
        x="r_type",
        y="mean_pCorr",
        hue="sac_dir",
        order=["stim", "choice"],
    )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()

    print("Group difference unsing independent t-test:")
    print(ttest_ind(stim_diff, choice_diff))
    print("Group difference using paired t-test:")
    print(ttest_rel(stim_diff, choice_diff))

    f.close()

    ax.set_xticklabels(["R-stimulus", "R-choice"])
    ax.set(ylabel="Average r", xlabel="")
    ax.tick_params(bottom=True, left=True)
    plt.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(plt_dir, "pCorr_net_comp.pdf"), format="pdf")
        plt.savefig(os.path.join(plt_dir, "pCorr_net_comp.eps"), format="eps")
        plt.savefig(os.path.join(plt_dir, "pCorr_net_comp.png"), format="png")


def extract_pCorr_data(
    rep,
    motion_rng,
    m1,
    m_sel,
    shuf=None,
    coh_code={"H": 4, "M": 2, "L": 1, "Z": 0},
    pref_sign=1,
    non_sign=-1,
):
    if shuf is None:
        n = SimpleNamespace(
            **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
        )
    else:
        n = SimpleNamespace(
            **load_test_data(
                f_dir, "test_output_lr%f_rep%d_shuf%d.h5" % (lr, rep, shuf)
            )
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
    h_temp = h[50:, :, motion_rng][:, :, m_sel.astype(bool)]
    h_final = np.mean(h_temp, axis=0)

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
        if shuf is None:
            savemat(
                os.path.join(
                    f_dir, "pCorr_data", "single_pCorr_data_rep%d_m1.mat" % rep
                ),
                data_dict,
            )
        else:
            savemat(
                os.path.join(
                    f_dir,
                    "pCorr_data",
                    "single_pCorr_data_rep%d_shuf%d_m1.mat" % (rep, shuf),
                ),
                data_dict,
            )
    else:
        if shuf is None:
            savemat(
                os.path.join(
                    f_dir, "pCorr_data", "single_pCorr_data_rep%d_m2.mat" % rep
                ),
                data_dict,
            )
        else:
            savemat(
                os.path.join(
                    f_dir,
                    "pCorr_data",
                    "single_pCorr_data_rep%d_shuf%d_m2.mat" % (rep, shuf),
                ),
                data_dict,
            )


main()
