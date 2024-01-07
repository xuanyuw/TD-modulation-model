import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import (
    find_coh_idx,
    find_sac_idx,
    combine_idx,
    load_test_data,
    get_pref_idx,
    pick_selective_neurons,
)
from types import SimpleNamespace
from pickle import dump, load
from tqdm import tqdm
from time import perf_counter
from scipy.stats import ttest_rel, sem, ttest_1samp
from joblib import Parallel, delayed
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator

# plot settings

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.figsize"] = [6, 4]


f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    # "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_shufFeedback_model",
    # "cutSpec_model",
    # "cutNonspec_model",
]

network_names = ["Full", "noFeedback", "shufFeedback", "CutSpec", "CutNonspec"]

# f_dirs = [
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
# ]
# network_names = [
#     "Full",
#     "noFeedback",
# ]

data_dir = "F:\\Github\\TD-modulation-model\\generate_figs\\nl_mod_idx_data"
plot_dir = "F:\\Github\\TD-modulation-model\\generate_figs\\nl_mod_idx_figs"

total_rep = 50
all_rep = range(total_rep)

true_neuron = False
stim_st_time = 45
target_st_time = 25
rerun_calc = False
plot_sel = True
save_plot = True

calc_range = [int(stim_st_time + (100 / 20)), int(stim_st_time + (300 / 20))]

if os.path.exists(data_dir) == False:
    os.mkdir(data_dir)
if os.path.exists(plot_dir) == False:
    os.mkdir(plot_dir)


def main():
    # load data

    mod_idx = calc_all_mod_idx_unit(f_dirs, data_dir, all_rep, rerun_calc, calc_range)

    # plot modulation index
    # plot example network distribution
    # example_net_mod_idx = [mod_idx[0][0], mod_idx[1][0], mod_idx[2][0]]
    # plot_mod_idx(example_net_mod_idx, network_names, save_plot, True)

    # plot all network distribution
    all_net_mod_idx = []
    shuf_idx = np.where(["shuf" in fd for fd in f_dirs])[0]
    for i in range(len(mod_idx)):
        if i not in shuf_idx:
            all_net_mod_idx.append(np.array([np.mean(x, axis=0) for x in mod_idx[i]]))
        else:
            all_net_mod_idx.append(np.array(mod_idx[i]))
    print(ttest_rel(all_net_mod_idx[0][:, 0], all_net_mod_idx[1]))
    # plot_mod_idx(all_net_mod_idx, network_names, save_plot, False)
    # plot_mod_idx_all_coh(all_net_mod_idx, network_names, save_plot)
    return


def calc_all_mod_idx_unit(f_dirs, data_dir, all_rep, rerun_calc, calc_range):
    """
    the return data is a 2D list with shape net x rep, each element is a matrix
    with shape neuron x coh, the coh contains 5 levels: all, H, M, L, and Z
    if shuf is in the f_dir, then each element is a vector with length 5, which si the mean of all shuffles all units in a rep
    """
    if rerun_calc:
        all_mod_idx = []
        for f_dir in f_dirs:
            model_mod_idx = []
            if "shuf" in f_dir:
                for rep in tqdm(all_rep):
                    rep_mod_idx = np.empty((100))
                    for shuf in range(100):
                        # load data
                        n = SimpleNamespace(
                            **load_test_data(
                                f_dir,
                                "test_output_lr%f_rep%d_shuf%d.h5" % (0.02, rep, shuf),
                            )
                        )
                        h = n.h
                        y = n.y

                        stim_dir = n.stim_dir
                        coh = n.stim_level
                        coh_idx = find_coh_idx(coh)
                        m1_rng = np.concatenate((np.arange(0, 40), np.arange(160, 170)))
                        m2_rng = np.concatenate(
                            (np.arange(80, 120), np.arange(180, 190))
                        )
                        if plot_sel:
                            sel_idx = pick_selective_neurons(h, n.stim_dir)
                            sel_idx = np.where(sel_idx == 1)[0]
                            m1_rng = np.intersect1d(m1_rng, sel_idx)
                            m2_rng = np.intersect1d(m2_rng, sel_idx)
                        num_cell = len(m1_rng) + len(m2_rng)

                        # shuf_mod_idx = np.empty(5)
                        mod_idx_m1 = calc_mod_idx_unit(
                            h, y, stim_dir, None, m1_rng, calc_range, True
                        )
                        mod_idx_m2 = calc_mod_idx_unit(
                            h, y, stim_dir, None, m2_rng, calc_range, False
                        )
                        shuf_mod_idx = np.mean(np.concatenate((mod_idx_m1, mod_idx_m2)))
                        rep_mod_idx[shuf] = shuf_mod_idx
                        if np.isnan(np.mean(rep_mod_idx)):
                            print("nan at rep %d" % rep)
                    model_mod_idx.append(np.mean(rep_mod_idx))
            else:
                for rep in tqdm(all_rep):
                    # load data
                    n = SimpleNamespace(
                        **load_test_data(
                            f_dir, "test_output_lr%f_rep%d.h5" % (0.02, rep)
                        )
                    )
                    h = n.h
                    y = n.y

                    stim_dir = n.stim_dir
                    coh = n.stim_level
                    coh_idx = find_coh_idx(coh)
                    m1_rng = np.concatenate((np.arange(0, 40), np.arange(160, 170)))
                    m2_rng = np.concatenate((np.arange(80, 120), np.arange(180, 190)))
                    if plot_sel:
                        sel_idx = pick_selective_neurons(h, n.stim_dir)
                        sel_idx = np.where(sel_idx == 1)[0]
                        m1_rng = np.intersect1d(m1_rng, sel_idx)
                        m2_rng = np.intersect1d(m2_rng, sel_idx)
                    num_cell = len(m1_rng) + len(m2_rng)

                    rep_mod_idx = np.empty((num_cell, 5))
                    mod_idx_m1 = calc_mod_idx_unit(
                        h, y, stim_dir, None, m1_rng, calc_range, True
                    )
                    mod_idx_m2 = calc_mod_idx_unit(
                        h, y, stim_dir, None, m2_rng, calc_range, False
                    )
                    rep_mod_idx[:, 0] = np.concatenate((mod_idx_m1, mod_idx_m2))

                    # rep_mod_idx[:, 0] = calc_mod_idx(avg_h_c0.reshape(1, -1), avg_h_c1.reshape(1, -1))
                    for c_idx, c in enumerate(["H", "M", "L", "Z"]):
                        mod_idx_m1 = calc_mod_idx_unit(
                            h, y, stim_dir, coh_idx[c], m1_rng, calc_range, True
                        )
                        mod_idx_m2 = calc_mod_idx_unit(
                            h, y, stim_dir, coh_idx[c], m2_rng, calc_range, False
                        )
                        rep_mod_idx[:, c_idx + 1] = np.concatenate(
                            (mod_idx_m1, mod_idx_m2)
                        )

                    model_mod_idx.append(rep_mod_idx)
            all_mod_idx.append(model_mod_idx)
        # save data
        with open(os.path.join(data_dir, "all_network_mod_idx.pkl"), "wb") as f:
            dump(all_mod_idx, f)
    else:
        with open(os.path.join(data_dir, "all_network_mod_idx.pkl"), "rb") as f:
            all_mod_idx = load(f)

    return all_mod_idx


def calc_mod_idx_unit(h, y, stim_dir, coh, m_rng, calc_range, m1):
    # NMI = 0 when contra v.s. ipsi selectivity are the same, =1 when contra selectivity and ipsi selectivity are different (one larger)
    contra_idx, ipsi_idx = find_sac_idx(y, m1)
    contra_dir135_idx = combine_idx(contra_idx, stim_dir == 135, coh)
    contra_dir315_idx = combine_idx(contra_idx, stim_dir == 315, coh)
    ipsi_dir135_idx = combine_idx(ipsi_idx, stim_dir == 135, coh)
    ipsi_dir315_idx = combine_idx(ipsi_idx, stim_dir == 315, coh)

    if (
        sum(contra_dir135_idx) == 0
        or sum(contra_dir315_idx) == 0
        or sum(ipsi_dir135_idx) == 0
        or sum(ipsi_dir315_idx) == 0
    ):
        return np.nan * np.ones(len(m_rng))

    h_contra_d1 = h[calc_range[0] : calc_range[1], contra_dir135_idx, :][:, :, m_rng]
    h_contra_d2 = h[calc_range[0] : calc_range[1], contra_dir315_idx, :][:, :, m_rng]
    h_ipsi_d1 = h[calc_range[0] : calc_range[1], ipsi_dir135_idx, :][:, :, m_rng]
    h_ipsi_d2 = h[calc_range[0] : calc_range[1], ipsi_dir315_idx, :][:, :, m_rng]

    avg_contra_d1 = np.mean(h_contra_d1, axis=(0, 1), where=(~np.isnan(h_contra_d1)))
    avg_contra_d2 = np.mean(h_contra_d2, axis=(0, 1), where=(~np.isnan(h_contra_d2)))
    avg_ipsi_d1 = np.mean(h_ipsi_d1, axis=(0, 1), where=(~np.isnan(h_ipsi_d1)))
    avg_ipsi_d2 = np.mean(h_ipsi_d2, axis=(0, 1), where=(~np.isnan(h_ipsi_d2)))

    A = avg_contra_d1 - avg_ipsi_d1
    B = avg_contra_d2 - avg_ipsi_d2

    mod_idx = np.abs(A - B) / (np.abs(A) + np.abs(B))

    return mod_idx


def plot_mod_idx_all_coh(mod_idx, net_names, save_plot):
    data_df = build_dataframe(mod_idx, net_names)
    plot_data_df = data_df[data_df["coherence"] == "all"]
    fig, ax = plt.subplots(1, 1)
    sns.stripplot(
        x="network", y="mod_idx", hue="network", dodge=True, data=plot_data_df, ax=ax
    )
    # ax.legend()
    ax.set_xlabel("Coherence")
    ax.set_ylabel("NMI")

    pairs = [
        ("Full", "noFeedback"),
        ("Full", "shufFeedback"),
        ("Full", "CutSpec"),
        ("Full", "CutNonspec"),
    ]

    fn = "stat_test_all_net_all_coh.txt"
    f = open(os.path.join(plot_dir, fn), "w")
    sys.stdout = f

    # perform two-way ANOVA on full and noFeedback model
    temp_df = data_df[data_df["coherence"] != "all"]
    calc_ANOVA2(temp_df, ["Full", "noFeedback"])

    annot = Annotator(
        ax,
        pairs,
        data=plot_data_df,
        x="network",
        y="mod_idx",
    )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()
    plt.tight_layout()

    if save_plot:
        plt.savefig(os.path.join(plot_dir, "all_network_mod_idx_all_coh.pdf"))
    return


def calc_ANOVA2(df, model_names):
    # the df should not contain 'all' coherence
    temp_df = df[(df["network"] == model_names[0]) | (df["network"] == model_names[1])]
    model = ols(
        "mod_idx ~ C(network) + C(coherence) +\
    C(network):C(coherence)",
        data=temp_df[temp_df["coherence"] != "Z"],
    ).fit()
    result = sm.stats.anova_lm(model, type=2)
    print("\n")
    print(
        "Two-way ANOVA compare %s v.s. %s Results:" % (model_names[0], model_names[1])
    )
    print(result)


def plot_mod_idx(mod_idx, net_names, save_plot, single_cell):
    """
    The input mod_idx has to be a list with number of element = number of networks
    """

    data_df = build_dataframe(mod_idx, net_names)
    fig, ax = plt.subplots(1, 1)
    sns.stripplot(
        x="coherence", y="mod_idx", hue="network", dodge=True, data=data_df, ax=ax
    )
    ax.legend()
    ax.set_xlabel("Coherence")
    ax.set_ylabel("NMI")

    pairs = [
        (("all", "Full"), ("all", "CutSpec")),
        (("all", "Full"), ("all", "CutNonspec")),
        (("H", "Full"), ("H", "CutSpec")),
        (("H", "Full"), ("H", "CutNonspec")),
        (("M", "Full"), ("M", "CutSpec")),
        (("M", "Full"), ("M", "CutNonspec")),
        (("L", "Full"), ("L", "CutSpec")),
        (("L", "Full"), ("L", "CutNonspec")),
        (("Z", "Full"), ("Z", "CutSpec")),
        (("Z", "Full"), ("Z", "CutNonspec")),
    ]

    if single_cell:
        fn = "stat_test_example_net.txt"
    else:
        fn = "stat_test_all_net.txt"
    f = open(os.path.join(plot_dir, fn), "w")
    sys.stdout = f

    annot = Annotator(
        ax,
        pairs,
        data=data_df,
        x="coherence",
        y="mod_idx",
        hue="network",
    )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()
    plt.tight_layout()

    if single_cell:
        if save_plot:
            plt.savefig(os.path.join(plot_dir, "example_network_mod_idx.pdf"))
    else:
        if save_plot:
            plt.savefig(os.path.join(plot_dir, "all_network_mod_idx.pdf"))

    return


def build_dataframe(mod_idx, net_names):
    """
    The input mod_idx has to be a list with number of element = number of networks
    """
    df = pd.DataFrame()
    for i, m in enumerate(mod_idx):
        num_rep = m.shape[0]
        coh_names = ["all", "H", "M", "L", "Z"]
        for j in range(len(coh_names)):
            df = df.append(
                pd.DataFrame(
                    {
                        "mod_idx": m[:, j],
                        "network": net_names[i],
                        "coherence": coh_names[j],
                    }
                )
            )
    return df


if __name__ == "__main__":
    main()
