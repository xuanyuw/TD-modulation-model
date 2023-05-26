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
from plot_ROC import get_sel_cells
import seaborn as sns
from matplotlib.colors import to_rgb
from types import SimpleNamespace
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from statannotations.Annotator import Annotator
from plot_ROC import rocN
from utils import (
    find_sac_idx,
    combine_idx,
    load_test_data,
    min_max_normalize,
    find_coh_idx,
    get_pref_idx,
)
from tqdm import tqdm
from time import perf_counter
from joblib import Parallel, delayed
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_1samp

# plot settings

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2


# f_dirs = [
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_shufFeedback_model",
# ]
f_dirs = ["crossOutput_noInterneuron_noMTConn_removeFB_model"]

# f_dirs = ["cutSpec_model", "cutNonspec_model"]

# plt_dir = os.path.join("generate_figs", "ROC_population_plots")
# data_dir = os.path.join("generate_figs", "ROC_population_data")

plt_dir = os.path.join("generate_figs", "rmv_fb_plots", "ROC_population_plots")
data_dir = os.path.join("generate_figs", "rmv_fb_plots", "ROC_population_data")

# plt_dir = os.path.join("generate_figs", "cut_fb_plots", "ROC_population_plots")
# data_dir = os.path.join("generate_figs", "cut_fb_plots", "ROC_population_data")

if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

total_rep = 50
total_shuf = 100
# total_shuf = 1
# f_dir = "/Users/xuanyuwu/Documents/GitHub/TD-modulation-model/crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
all_rep = range(total_rep)
# all_rep = range(27,total_rep)
lr = 0.02

stim_st_time = 45
target_st_time = 25
normalize = True
plot_sel = True
rerun_calc = False
save_plot = False

n_jobs = 8


def main():
    sep_sac_df, comb_sac_df = load_roc()

    f = open(os.path.join(plt_dir, "stat_test.txt"), "w")
    sys.stdout = f

    for model_type in ["Remove feedback"]:
        # for model_type in ["cutSpec", "cutNonspec"]:
        # for model_type in ["Full model", "No feedback"]:  # , "Shuffled feedback"]:
        print("%s Sep Sac Stats" % model_type)
        df = sep_sac_df[sep_sac_df["model"] == model_type]
        df = df.groupby(["rep", "coh", "sac"]).mean().reset_index()

        plot_violin_sep_sac(df, model_type)
        print("---------------------------")
        print("\n")

    # print("Dir Sel Comb Sac Stats")
    # plot_violin(comb_sac_df[comb_sac_df["type"] == "dir"], "dir")

    # print("---------------------------")
    # print("\n")
    # print("Sac Sel Comb Sac Stats")
    # plot_violin(comb_sac_df[comb_sac_df["type"] == "sac"], "sac")
    # print("---------------------------")
    # print("\n")

    # print("Ipsi v.s. Contra Diff Comparison (Full vs. Shuf")
    # plot_sac_roc_diff(sep_sac_df)

    f.close()


def plot_violin(df, roc_type):
    fig, ax = plt.subplots()
    # color_palette = {'H': '#FF0000', 'M': '#00FF00', 'L':'#0000FF', 'Z': 'k'}
    colors = ["#FF0000", "#00FF00", "#0000FF", "#424242"]
    handles = []
    sns.violinplot(
        x="coh",
        y="roc",
        hue="model",
        data=df.astype({"roc": float}),
        inner="points",
        ax=ax,
        palette=colors,
        hue_order=["Full model", "No feedback", "Shuffled feedback"],
        order=["H", "M", "L", "Z"],
    )

    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 3])
        if ind % 3 == 1:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        if ind % 3 == 2:
            rgb = 0.8 + 0.2 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor="black"))

    ax.legend(
        handles=[tuple(handles[::3]), tuple(handles[1::3]), tuple(handles[2::3])],
        labels=df["model"].astype("category").cat.categories.to_list(),
        handlelength=4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="best",
        frameon=False,
    )

    # add statistical test results
    pairs = [
        (("H", "Full model"), ("H", "No feedback")),
        (("H", "Full model"), ("H", "Shuffled feedback")),
        (("M", "Full model"), ("M", "No feedback")),
        (("M", "Full model"), ("M", "Shuffled feedback")),
        (("L", "Full model"), ("L", "No feedback")),
        (("L", "Full model"), ("L", "Shuffled feedback")),
        (("Z", "Full model"), ("Z", "No feedback")),
        (("Z", "Full model"), ("Z", "Shuffled feedback")),
    ]

    annot = Annotator(
        ax, pairs, data=df, x="coh", y="roc", hue="model", order=["H", "M", "L", "Z"]
    )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()

    ax.set(xlabel="Coherence", ylabel="Average ROC")
    ax.tick_params(bottom=True, left=True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(plt_dir, "%s_roc_net_comp_comb_sac.pdf" % roc_type), format="pdf"
    )
    plt.savefig(
        os.path.join(plt_dir, "%s_roc_net_comp_comb_sac.eps" % roc_type), format="eps"
    )
    plt.savefig(
        os.path.join(plt_dir, "%s_roc_net_comp_comb_sac.png" % roc_type), format="png"
    )

    df_mean = df[["model", "coh", "roc"]].groupby(["model", "coh"]).mean()
    df_mean.to_csv(os.path.join(plt_dir, "all_model_coh_%s_roc_mean.csv" % roc_type))

    # calculate anova
    print("\n")
    for model in df["model"].unique():
        temp_df = df[df["model"] == model]
        print(model + "one-way ANOVA result:")
        oneway_result = f_oneway(
            temp_df[temp_df["coh"] == "H"]["roc"],
            temp_df[temp_df["coh"] == "M"]["roc"],
            temp_df[temp_df["coh"] == "L"]["roc"],
            temp_df[temp_df["coh"] == "Z"]["roc"],
        )
        # oneway_result = f_oneway(temp_df[temp_df['coh']=='H']['roc'], temp_df[temp_df['coh']=='M']['roc'],
        # temp_df[temp_df['coh']=='L']['roc'])
        print(oneway_result)

        print(model + " linear regression result:")
        print(calc_linear_corr_sig(temp_df))

    for m in ["Shuffled feedback", "No feedback"]:
        temp_df = df[(df["model"] == "Full model") | (df["model"] == m)]
        model = ols(
            "roc ~ C(model) + C(coh) +\
        C(model):C(coh)",
            data=temp_df[temp_df["coh"] != "Z"],
        ).fit()
        twoway_result = sm.stats.anova_lm(model, type=2)
        print("\n")
        print("Two-way ANOVA compare %s Results:" % m)
        print(twoway_result)


def calc_linear_corr_sig(df):
    r_arr = []

    def get_roc(temp, coh):
        return temp[temp.coh == coh].roc.to_numpy()[0]

    for rep in df["rep"].unique():
        temp = df[df["rep"] == rep]
        x = np.array([0.75, 0.55, 0.35, 0]).reshape((-1, 1))
        y = np.array(
            [
                get_roc(temp, "H"),
                get_roc(temp, "M"),
                get_roc(temp, "L"),
                get_roc(temp, "Z"),
            ]
        )
        model = LinearRegression().fit(x, y)
        r_arr.append(model.coef_)
    print("Mean slop = %.4f" % np.mean(r_arr))
    return ttest_1samp(r_arr, 0)


def plot_violin_sep_sac(df, model_type):
    # calculate anova
    print("\n")
    print("%s two-way ANOVA result:" % model_type)
    model = ols(
        "roc ~ C(sac) + C(coh) +\
    C(sac):C(coh)",
        data=df[df["coh"] != "Z"],
    ).fit()
    twoway_result = sm.stats.anova_lm(model, type=2)
    print("\n")
    print("Two-way ANOVA Results:")
    print(twoway_result)

    fig, ax = plt.subplots()
    # color_palette = {'H': '#FF0000', 'M': '#00FF00', 'L':'#0000FF', 'Z': 'k'}
    colors = ["#FF0000", "#00FF00", "#0000FF", "#424242"]
    handles = []
    sns.violinplot(
        x="coh",
        y="roc",
        hue="sac",
        data=df.astype({"roc": float}),
        inner="quartile",
        ax=ax,
        palette=[".2", ".5"],
        hue_order=["ipsi", "contra"],
        split=True,
        order=["H", "M", "L", "Z"],
    )

    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 1:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor="black"))

    ax.legend(
        handles=[tuple(handles[1::2]), tuple(handles[::2])],
        labels=df["sac"].astype("category").cat.categories.to_list(),
        handlelength=4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="lower left",
        frameon=False,
    )

    # add statistical test results
    pairs = [
        (("H", "ipsi"), ("H", "contra")),
        (("M", "ipsi"), ("M", "contra")),
        (("L", "ipsi"), ("L", "contra")),
        (("Z", "ipsi"), ("Z", "contra")),
    ]

    annot = Annotator(
        ax, pairs, data=df, x="coh", y="roc", hue="sac", order=["H", "M", "L", "Z"]
    )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()

    ax.set(xlabel="Coherence", ylabel="Average ROC")
    ax.tick_params(bottom=True, left=True)
    plt.tight_layout()
    if save_plot:
        plt.savefig(
            os.path.join(plt_dir, "dir_sel_net_comp_%s.pdf" % model_type), format="pdf"
        )
        plt.savefig(
            os.path.join(plt_dir, "dir_sel_net_comp_%s.eps" % model_type), format="eps"
        )
        plt.savefig(
            os.path.join(plt_dir, "dir_sel_net_comp_%s.png" % model_type), format="png"
        )

    df_mean = df[["sac", "coh", "roc"]].groupby(["sac", "coh"]).mean()
    df_mean.to_csv(
        os.path.join(plt_dir, "all_model_sep_sac_roc_mean_%s.csv" % model_type)
    )


def plot_sac_roc_diff(df):
    df = df[(df["model"] == "Full model") | (df["model"] == "Shuffled feedback")]
    temp_idx = df["sac"] == "contra"
    temp_df = df[temp_idx]
    roc_diff = df[df["sac"] == "contra"]["roc"].reset_index(drop=True) - df[
        df["sac"] == "ipsi"
    ]["roc"].reset_index(drop=True)
    new_df = pd.DataFrame(
        {
            "rep": temp_df["rep"].reset_index(drop=True),
            "coh": temp_df["coh"].reset_index(drop=True),
            "model": temp_df["model"].reset_index(drop=True),
            "roc_diff": roc_diff,
        }
    )

    fig, ax = plt.subplots()
    colors = ["#FF0000", "#00FF00", "#0000FF", "#424242"]
    sns.stripplot(
        x="coh",
        y="roc_diff",
        hue="model",
        data=new_df,
        ax=ax,
        palette=[".2", ".5"],
        dodge=True,
        hue_order=["Full model", "Shuffled feedback"],
        order=["H", "M", "L", "Z"],
    )
    ax.legend(loc="best", frameon=False)

    # sns.catplot(x = 'coh_model', y = 'roc_diff', data = new_df, ax=ax,
    #             palette=['.2', '.5'],
    #             order=['H, Full model', 'H, Shuffled feedback', 'M, Full model', 'M, Shuffled feedback', 'L, Full model', 'L, Shuffled feedback', 'Z, Full model', 'Z, Shuffled feedback'])

    # add statistical test results
    pairs = [
        (("H", "Full model"), ("H", "Shuffled feedback")),
        (("M", "Full model"), ("M", "Shuffled feedback")),
        (("L", "Full model"), ("L", "Shuffled feedback")),
        (("Z", "Full model"), ("Z", "Shuffled feedback")),
    ]

    annot = Annotator(
        ax,
        pairs,
        data=new_df,
        x="coh",
        y="roc_diff",
        hue="model",
        order=["H", "M", "L", "Z"],
    )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()

    ax.set(xlabel="Coherence", ylabel="AUC Differences")
    ax.tick_params(bottom=True, left=True)
    plt.tight_layout()
    if save_plot:
        plt.savefig(
            os.path.join(plt_dir, "dir_sel_diff_full_vs_shuf.pdf"), format="pdf"
        )
        plt.savefig(
            os.path.join(plt_dir, "dir_sel_diff_full_vs_shuf.eps"), format="eps"
        )
        plt.savefig(
            os.path.join(plt_dir, "dir_sel_diff_full_vs_shuf.png"), format="png"
        )


def load_single_file_ROC(
    f_dir,
    rep,
    motion_rng,
    m1_rng,
    motion_idx,
    target_idx,
    motion_selective,
    saccade_selective,
    shuf=None,
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

    coh_dict = find_coh_idx(n.stim_level)
    pref_dir, pref_sac = get_pref_idx(n, h)

    m_sel = motion_selective[rep].astype(bool)
    s_sel = saccade_selective[rep].astype(bool)

    m_sel_rng = combine_idx(m_sel, motion_idx)
    s_sel_rng = combine_idx(s_sel, target_idx)

    temp_sep_sac_df = pd.DataFrame(columns=["rep", "shuf", "coh", "sac", "roc"])
    temp_comb_sac_df = pd.DataFrame(columns=["rep", "shuf", "coh", "type", "roc"])

    for coh in ["H", "M", "L", "Z"]:
        ipsi_roc, contra_roc = calc_sac_sep_ROC(
            h[50:, :, motion_rng], n, m1_rng, coh_dict[coh], pref_dir[:, motion_rng]
        )

        dir_roc = calc_ROC(
            h[50:, :, m_sel_rng], n, coh_dict[coh], pref_dir[:, m_sel_rng]
        )
        sac_roc = calc_ROC(
            h[60:, :, s_sel_rng], n, coh_dict[coh], pref_sac[:, s_sel_rng]
        )  # use 100ms before and after mean rt to calculate saccade sel

        # populate df
        rep_arr = [rep] * 2
        shuf_arr = [shuf] * 2
        coh_arr = [coh] * 2
        sac_arr = ["ipsi", "contra"]
        type_arr = ["dir", "sac"]
        sep_sac_arr = [ipsi_roc, contra_roc]
        comb_sac_arr = [dir_roc, sac_roc]
        temp_sep_sac_df = pd.concat(
            (
                temp_sep_sac_df,
                pd.DataFrame(
                    {
                        "rep": rep_arr,
                        "shuf": shuf_arr,
                        "coh": coh_arr,
                        "sac": sac_arr,
                        "roc": sep_sac_arr,
                    }
                ),
            )
        )
        temp_comb_sac_df = pd.concat(
            (
                temp_comb_sac_df,
                pd.DataFrame(
                    {
                        "rep": rep_arr,
                        "shuf": shuf_arr,
                        "coh": coh_arr,
                        "type": type_arr,
                        "roc": comb_sac_arr,
                    }
                ),
            )
        )
    return temp_sep_sac_df, temp_comb_sac_df


def calc_all_ROC():
    motion_rng = np.concatenate(
        (np.arange(0, 40), np.arange(80, 120), np.arange(160, 170), np.arange(180, 190))
    )
    motion_idx = np.zeros((200,)).astype(bool)
    motion_idx[motion_rng] = True
    target_rng = np.concatenate(
        (
            np.arange(40, 80),
            np.arange(120, 160),
            np.arange(170, 180),
            np.arange(190, 200),
        )
    )
    target_idx = np.zeros((200,)).astype(bool)
    target_idx[target_rng] = True
    m1_rng = np.concatenate(
        (np.arange(0, 40), np.arange(80, 90))
    )  # range of m1 neuron indices after separated by RF

    motion_selective, saccade_selective = get_sel_cells()
    for f_dir in f_dirs:
        if "shuf" not in f_dir:
            pbar = tqdm(total=total_rep)
            sep_sac_roc_df = pd.DataFrame(columns=["rep", "shuf", "coh", "sac", "roc"])
            comb_sac_roc_df = pd.DataFrame(
                columns=["rep", "shuf", "coh", "type", "roc"]
            )
            for rep in all_rep:
                temp_sep_sac_df, temp_comb_sac_df = load_single_file_ROC(
                    f_dir,
                    rep,
                    motion_rng,
                    m1_rng,
                    motion_idx,
                    target_idx,
                    motion_selective,
                    saccade_selective,
                )
                if "noFeedback" in f_dir:
                    temp_sep_sac_df["model"] = ["No feedback"] * len(
                        temp_sep_sac_df.index
                    )
                    temp_comb_sac_df["model"] = ["No feedback"] * len(
                        temp_comb_sac_df.index
                    )
                elif "removeFB" in f_dir:
                    temp_sep_sac_df["model"] = ["Remove feedback"] * len(
                        temp_sep_sac_df.index
                    )
                    temp_comb_sac_df["model"] = ["Remove feedback"] * len(
                        temp_comb_sac_df.index
                    )
                elif "cutSpec" in f_dir:
                    temp_sep_sac_df["model"] = ["Cut spec"] * len(temp_sep_sac_df.index)
                    temp_comb_sac_df["model"] = ["Cut spec"] * len(
                        temp_comb_sac_df.index
                    )
                elif "cutNonspec" in f_dir:
                    temp_sep_sac_df["model"] = ["Cut nonspec"] * len(
                        temp_sep_sac_df.index
                    )
                    temp_comb_sac_df["model"] = ["Cut nonspec"] * len(
                        temp_comb_sac_df.index
                    )
                else:
                    temp_sep_sac_df["model"] = ["Full model"] * len(
                        temp_sep_sac_df.index
                    )
                    temp_comb_sac_df["model"] = ["Full model"] * len(
                        temp_comb_sac_df.index
                    )
                sep_sac_roc_df = pd.concat([sep_sac_roc_df, temp_sep_sac_df])
                comb_sac_roc_df = pd.concat([comb_sac_roc_df, temp_comb_sac_df])
                pbar.update(1)

            sep_sac_roc_df = sep_sac_roc_df.explode("roc").reset_index()
            comb_sac_roc_df = comb_sac_roc_df.explode("roc").reset_index()
            if "noFeedback" in f_dir:
                sep_sac_roc_df.to_csv(
                    os.path.join(data_dir, "noFeedback_all_nets_ROC_sep_sac.csv")
                )
                comb_sac_roc_df.to_csv(
                    os.path.join(data_dir, "noFeedback_all_nets_ROC_comb_sac.csv")
                )
            elif "removeFB" in f_dir:
                sep_sac_roc_df.to_csv(
                    os.path.join(data_dir, "removeFB_all_nets_ROC_sep_sac.csv")
                )
                comb_sac_roc_df.to_csv(
                    os.path.join(data_dir, "removeFB_all_nets_ROC_comb_sac.csv")
                )
            elif "cutSpec" in f_dir:
                sep_sac_roc_df.to_csv(
                    os.path.join(data_dir, "cutSpec_all_nets_ROC_sep_sac.csv")
                )
                comb_sac_roc_df.to_csv(
                    os.path.join(data_dir, "cutSpec_all_nets_ROC_comb_sac.csv")
                )
            elif "cutNonspec" in f_dir:
                sep_sac_roc_df.to_csv(
                    os.path.join(data_dir, "cutNonspec_all_nets_ROC_sep_sac.csv")
                )
                comb_sac_roc_df.to_csv(
                    os.path.join(data_dir, "cutNonspec_all_nets_ROC_comb_sac.csv")
                )
            else:
                sep_sac_roc_df.to_csv(
                    os.path.join(data_dir, "fullModel_all_nets_ROC_sep_sac.csv")
                )
                comb_sac_roc_df.to_csv(
                    os.path.join(data_dir, "fullModel_all_nets_ROC_comb_sac.csv")
                )
            pbar.close()
        else:
            pbar = tqdm(total=len(all_rep) * total_shuf)
            for rep in all_rep:
                sep_sac_roc_df = pd.DataFrame(
                    columns=["rep", "shuf", "coh", "sac", "roc"]
                )
                comb_sac_roc_df = pd.DataFrame(
                    columns=["rep", "shuf", "coh", "type", "roc"]
                )
                for shuf in range(total_shuf):
                    temp_sep_sac_df, temp_comb_sac_df = load_single_file_ROC(
                        f_dir,
                        rep,
                        motion_rng,
                        m1_rng,
                        motion_idx,
                        target_idx,
                        motion_selective,
                        saccade_selective,
                        shuf,
                    )
                    temp_sep_sac_df["model"] = ["Shuffled feedback"] * len(
                        temp_sep_sac_df.index
                    )
                    temp_comb_sac_df["model"] = ["Shuffled feedback"] * len(
                        temp_comb_sac_df.index
                    )

                    sep_sac_roc_df = pd.concat([sep_sac_roc_df, temp_sep_sac_df])
                    comb_sac_roc_df = pd.concat([comb_sac_roc_df, temp_comb_sac_df])
                    pbar.update(1)

                sep_sac_roc_df = sep_sac_roc_df.explode("roc").reset_index()
                comb_sac_roc_df = comb_sac_roc_df.explode("roc").reset_index()
                sep_sac_roc_df.to_csv(
                    os.path.join(
                        data_dir, "shufFeedback_all_nets_ROC_sep_sac_rep%d.csv" % rep
                    )
                )
                comb_sac_roc_df.to_csv(
                    os.path.join(
                        data_dir, "shufFeedback_all_nets_ROC_comb_sac_rep%d.csv" % rep
                    )
                )
            pbar.close()


def load_roc():
    if rerun_calc:
        # calc_all_ROC()

        sep_sac_df = pd.DataFrame(columns=["rep", "coh", "sac", "roc", "model"])
        comb_sac_df = pd.DataFrame(columns=["rep", "coh", "type", "roc", "model"])

        # mn = ["fullModel", "noFeedback"]  # , "shufFeedback"]
        mn = ["removeFB"]
        # mn = ["cutSpec", "cutNonspec"]
        for i in mn:
            if "shuf" in i:
                for rep in all_rep:
                    temp_sep_sac = (
                        pd.read_csv(
                            os.path.join(
                                data_dir, "%s_all_nets_ROC_sep_sac_rep%d.csv" % (i, rep)
                            )
                        )[["rep", "coh", "sac", "model", "roc"]]
                        .groupby(["rep", "coh", "sac", "model"])
                        .mean()
                        .reset_index()
                    )
                    temp_comb_sac = (
                        pd.read_csv(
                            os.path.join(
                                data_dir,
                                "%s_all_nets_ROC_comb_sac_rep%d.csv" % (i, rep),
                            )
                        )[["rep", "coh", "type", "model", "roc"]]
                        .groupby(["rep", "coh", "type", "model"])
                        .mean()
                        .reset_index()
                    )
                    sep_sac_df = pd.concat([sep_sac_df, temp_sep_sac])
                    comb_sac_df = pd.concat([comb_sac_df, temp_comb_sac])
            else:
                temp_sep_sac = (
                    pd.read_csv(
                        os.path.join(data_dir, "%s_all_nets_ROC_sep_sac.csv" % i)
                    )[["rep", "coh", "sac", "model", "roc"]]
                    .groupby(["rep", "coh", "sac", "model"])
                    .mean()
                    .reset_index()
                )
                temp_comb_sac = (
                    pd.read_csv(
                        os.path.join(data_dir, "%s_all_nets_ROC_comb_sac.csv" % i)
                    )[["rep", "coh", "type", "model", "roc"]]
                    .groupby(["rep", "coh", "type", "model"])
                    .mean()
                    .reset_index()
                )

                sep_sac_df = pd.concat([sep_sac_df, temp_sep_sac])
                comb_sac_df = pd.concat([comb_sac_df, temp_comb_sac])
        sep_sac_df.to_csv(os.path.join(data_dir, "net_avg_ROC_sep_sac.csv"))
        comb_sac_df.to_csv(os.path.join(data_dir, "net_avg_ROC_comb_sac.csv"))
    else:
        sep_sac_df = pd.read_csv(os.path.join(data_dir, "net_avg_ROC_sep_sac.csv"))
        comb_sac_df = pd.read_csv(os.path.join(data_dir, "net_avg_ROC_comb_sac.csv"))
    return sep_sac_df, comb_sac_df


def calc_sac_sep_ROC(h, n, m1_rng, coh_idx, pref_idx):
    contra_idx_m1, ipsi_idx_m1 = find_sac_idx(n.y, True)
    contra_idx_m2, ipsi_idx_m2 = find_sac_idx(n.y, False)

    ipsi_ROC = np.zeros((h.shape[2],))
    contra_ROC = np.zeros((h.shape[2],))

    def cell_roc_sep_sac(i):
        if i in m1_rng:
            contra_idx = contra_idx_m1
            ipsi_idx = ipsi_idx_m1
        else:
            contra_idx = contra_idx_m2
            ipsi_idx = ipsi_idx_m2
        ipsi_pref_idx = combine_idx(ipsi_idx, pref_idx[:, i], n.correct_idx, coh_idx)
        contra_pref_idx = combine_idx(
            contra_idx, pref_idx[:, i], n.correct_idx, coh_idx
        )
        ipsi_non_idx = combine_idx(ipsi_idx, ~pref_idx[:, i], n.correct_idx, coh_idx)
        contra_non_idx = combine_idx(
            contra_idx, ~pref_idx[:, i], n.correct_idx, coh_idx
        )

        h_ipsi_pref = np.mean(h[:, ipsi_pref_idx, i], axis=0)
        h_contra_pref = np.mean(h[:, contra_pref_idx, i], axis=0)
        h_ipsi_non = np.mean(h[:, ipsi_non_idx, i], axis=0)
        h_contra_non = np.mean(h[:, contra_non_idx, i], axis=0)

        if len(h_ipsi_pref) == 0 or len(h_ipsi_non) == 0:
            i_roc = np.nan
        else:
            i_roc = rocN(h_ipsi_pref, h_ipsi_non)
        if len(h_contra_pref) == 0 or len(h_contra_non) == 0:
            c_roc = np.nan
        else:
            c_roc = rocN(h_contra_pref, h_contra_non)

        return [i_roc, c_roc]

    # tic = perf_counter()
    return_li = Parallel(n_jobs=n_jobs)(
        delayed(cell_roc_sep_sac)(i) for i in range(h.shape[2])
    )
    # toc=perf_counter()
    # print(f"ROC ran in {toc - tic:0.4f} seconds")

    ipsi_ROC = np.array([i[0] for i in return_li])
    contra_ROC = np.array([i[1] for i in return_li])
    return ipsi_ROC, contra_ROC


def calc_ROC(h, n, coh_idx, pref_idx):
    # all_ROC = np.zeros((h.shape[2],))
    def cell_roc(i):
        pre_idx = combine_idx(pref_idx[:, i], n.correct_idx, coh_idx)
        non_idx = combine_idx(~pref_idx[:, i], n.correct_idx, coh_idx)

        h_pre = np.mean(h[:, pre_idx, i], axis=0)
        h_non = np.mean(h[:, non_idx, i], axis=0)

        if len(h_pre) == 0 or len(h_non) == 0:
            return np.nan
        return rocN(h_pre, h_non)

    # tic = perf_counter()
    all_ROC = Parallel(n_jobs=n_jobs)(delayed(cell_roc)(i) for i in range(h.shape[2]))
    # toc=perf_counter()
    # print(f"ROC ran in {toc - tic:0.4f} seconds")

    return np.array(all_ROC)


main()
