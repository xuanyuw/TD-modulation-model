import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pickle import load
import pandas as pd
import numpy as np
from plot_utils import calc_coh_rt
from statannotations.Annotator import Annotator
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import to_rgb
from tqdm import tqdm
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import f_oneway

# plot settings
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.figsize"] = [12, 5]

# f_dirs = [
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_shufFeedback_model",
# ]
f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    "cutSpec_model",
    "cutNonspec_model",
]
# f_dirs = [
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
#     "crossOutput_noInterneuron_noMTConn_removeFB_model",
# ]
# f_dirs = [
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
#     "trained_removeFB_model",
# ]
# plt_dir = os.path.join("generate_figs", "Fig7", "7d_ablation_rt_comp")
# plt_dir = os.path.join("generate_figs", "rmv_fb_plots", "rt_comp")
# plt_dir = os.path.join(
#     "generate_figs", "rmv_fb_plots", "rmv_fb_trained_eq_conn", "rt_comp"
# )
plt_dir = os.path.join("generate_figs", "cut_fb_plots", "rt_comp")
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)

total_rep = 50
total_shuf = 100
rerun_calculation = False


def main():
    df, below_thresh_count = load_rt()
    plot_violin(df)
    # plot_below_thresh_cnt(below_thresh_count)


def plot_below_thresh_cnt(below_thresh_count):
    below_thresh_count.columns = below_thresh_count.columns.droplevel(
        1
    )  # drop lambda multi-index
    _, ax = plt.subplots()
    colors = ["#FF0000", "#00FF00", "#0000FF", "#424242"]
    handles = []
    sns.violinplot(
        x="coh",
        y="rt",
        hue="model",
        data=below_thresh_count,
        inner="points",
        ax=ax,
        palette=[".2", ".5"],
        hue_order=["Full model", "No feedback", "Shuffled feedback"],
        order=["H", "M", "L", "Z"],
    )
    # sns.violinplot(
    #     x="coh",
    #     y="rt",
    #     hue="model",
    #     data=below_thresh_count,
    #     inner="points",
    #     ax=ax,
    #     palette=[".2", ".5"],
    #     hue_order=["Full model", "Remove feedback"],
    #     order=["H", "M", "L", "Z"],
    # )

    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 3])
        if ind % 3 == 1:
            rgb = 0.4 + 0.6 * np.array(rgb)  # make whiter
        if ind % 3 == 2:
            rgb = 0.7 + 0.3 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor="black"))

    ax.legend(
        handles=[tuple(handles[::3]), tuple(handles[1::3]), tuple(handles[2::3])],
        labels=below_thresh_count["model"].astype("category").cat.categories.to_list(),
        handlelength=4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="lower left",
        frameon=False,
    )

    # # add statistical test results
    # pairs = [(('H', 'Full model'), ('H', 'No feedback')), (('H', 'Full model'), ('H', 'Shuffled feedback')),
    #             (('M', 'Full model'), ('M', 'No feedback')), (('M', 'Full model'), ('M', 'Shuffled feedback')),
    #             (('L', 'Full model'), ('L', 'No feedback')), (('L', 'Full model'), ('L', 'Shuffled feedback')),
    #             (('Z', 'Full model'), ('Z', 'No feedback')), (('Z', 'Full model'), ('Z', 'Shuffled feedback'))]

    # f =  open(os.path.join(plt_dir, "stat_test.txt"), 'w')
    # sys.stdout = f

    # annot = Annotator(ax, pairs, data=below_thresh_count, x='coh', y='rt', hue='model', order=['H', 'M', 'L', 'Z'])
    # annot.configure(test='t-test_ind', text_format='star', loc='outside')
    # annot.apply_and_annotate()

    ax.set(xlabel="Coherence", ylabel="Count")
    ax.tick_params(bottom=True, left=True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(plt_dir, "ablation_rt_below_thresh_count.pdf"), format="pdf"
    )
    plt.savefig(
        os.path.join(plt_dir, "ablation_rt_below_thresh_count.eps"), format="eps"
    )
    plt.savefig(
        os.path.join(plt_dir, "ablation_rt_below_thresh_count.png"), format="png"
    )

    df_mean = (
        below_thresh_count[["model", "coh", "rt"]].groupby(["model", "coh"]).mean()
    )
    df_mean.to_csv(os.path.join(plt_dir, "below_thresh_count_mean.csv"))

    # # Performing two-way ANOVA
    # model = ols('rt ~ C(model) + C(coh) +\
    # C(model):C(coh)',
    #             data=df[df['coh']!='Z']).fit()
    # result = sm.stats.anova_lm(model, type=2)
    # print('\n')
    # print('Two-way ANOVA Result:')
    # print(result)

    # f.close()


def plot_violin(df):
    _, ax = plt.subplots()
    colors = ["#FF0000", "#00FF00", "#0000FF", "#424242"]
    handles = []
    # sns.violinplot(
    #     x="coh",
    #     y="rt",
    #     hue="model",
    #     data=df,
    #     inner="points",
    #     ax=ax,
    #     palette=[".2", ".5"],
    #     hue_order=["Full model", "No feedback", "Shuffled feedback"],
    #     order=["H", "M", "L", "Z"],
    # )
    # sns.violinplot(
    #     x="coh",
    #     y="rt",
    #     hue="model",
    #     data=df,
    #     inner="points",
    #     ax=ax,
    #     palette=[".2", ".5"],
    #     hue_order=["Full model", "Remove feedback"],
    #     order=["H", "M", "L", "Z"],
    # )
    sns.violinplot(
        x="coh",
        y="rt",
        hue="model",
        data=df,
        inner="points",
        ax=ax,
        palette=[".2", ".5"],
        hue_order=["Full model", "Cut Nonspecific", "Cut Specific"],
        order=["H", "M", "L", "Z"],
    )

    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 3])
        if ind % 3 == 1:
            rgb = 0.4 + 0.6 * np.array(rgb)  # make whiter
        if ind % 3 == 2:
            rgb = 0.7 + 0.3 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor="black"))
    # for ind, violin in enumerate(ax.findobj(PolyCollection)):
    #     rgb = to_rgb(colors[ind // 2])
    #     if ind % 2 == 1:
    #         rgb = 0.4 + 0.6 * np.array(rgb)  # make whiter
    #     violin.set_facecolor(rgb)
    #     handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor="black"))

    ax.legend(
        handles=[tuple(handles[::3]), tuple(handles[1::3]), tuple(handles[2::3])],
        labels=["Full model", "Cut Nonspecific", "Cut Specific"],
        # labels=df["model"].astype("category").cat.categories.to_list(),
        handlelength=4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="lower left",
        frameon=False,
    )

    # ax.legend(
    #     handles=[tuple(handles[::2]), tuple(handles[1::2])],
    #     labels=df["model"].astype("category").cat.categories.to_list(),
    #     handlelength=4,
    #     handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
    #     loc="lower left",
    #     frameon=False,
    # )

    # add statistical test results
    # pairs = [
    #     (("H", "Full model"), ("H", "No feedback")),
    #     (("H", "Full model"), ("H", "Shuffled feedback")),
    #     (("M", "Full model"), ("M", "No feedback")),
    #     (("M", "Full model"), ("M", "Shuffled feedback")),
    #     (("L", "Full model"), ("L", "No feedback")),
    #     (("L", "Full model"), ("L", "Shuffled feedback")),
    #     (("Z", "Full model"), ("Z", "No feedback")),
    #     (("Z", "Full model"), ("Z", "Shuffled feedback")),
    # ]
    # pairs = [
    #     (("H", "Full model"), ("H", "Remove feedback")),
    #     (("M", "Full model"), ("M", "Remove feedback")),
    #     (("L", "Full model"), ("L", "Remove feedback")),
    #     (("Z", "Full model"), ("Z", "Remove feedback")),
    # ]
    pairs = [
        (("H", "Full model"), ("H", "Cut Specific")),
        (("H", "Full model"), ("H", "Cut Nonspecific")),
        (("M", "Full model"), ("M", "Cut Specific")),
        (("M", "Full model"), ("M", "Cut Nonspecific")),
        (("L", "Full model"), ("L", "Cut Specific")),
        (("L", "Full model"), ("L", "Cut Nonspecific")),
        (("Z", "Full model"), ("Z", "Cut Specific")),
        (("Z", "Full model"), ("Z", "Cut Nonspecific")),
        (("H", "Cut Specific"), ("H", "Cut Nonspecific")),
        (("M", "Cut Specific"), ("M", "Cut Nonspecific")),
        (("L", "Cut Specific"), ("L", "Cut Nonspecific")),
        (("Z", "Cut Specific"), ("Z", "Cut Nonspecific")),
    ]

    f = open(os.path.join(plt_dir, "stat_test.txt"), "w")
    sys.stdout = f

    # annot = Annotator(
    #     ax,
    #     pairs,
    #     data=df,
    #     x="coh",
    #     y="rt",
    #     hue="model",
    #     order=["H", "M", "L", "Z"],
    #     hue_order=["Full model", "No feedback", "Shuffled feedback"],
    # )
    annot = Annotator(
        ax,
        pairs,
        data=df,
        x="coh",
        y="rt",
        hue="model",
        order=["H", "M", "L", "Z"],
        hue_order=["Full model", "Cut Nonspecific", "Cut Specific"],
    )
    # annot = Annotator(
    #     ax,
    #     pairs,
    #     data=df,
    #     x="coh",
    #     y="rt",
    #     hue="model",
    #     order=["H", "M", "L", "Z"],
    #     hue_order=["Full model", "Remove feedback"],
    # )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()

    ax.set(xlabel="Coherence", ylabel="Reaction Time (ms)")
    ax.tick_params(bottom=True, left=True)
    plt.tight_layout()

    plt.savefig(os.path.join(plt_dir, "ablation_rt_comp.pdf"), format="pdf")
    plt.savefig(os.path.join(plt_dir, "ablation_rt_comp.eps"), format="eps")
    plt.savefig(os.path.join(plt_dir, "ablation_rt_comp.png"), format="png")

    df_mean = df[["model", "coh", "rt"]].groupby(["model", "coh"]).mean()
    df_mean.to_csv(os.path.join(plt_dir, "all_model_rt_mean.csv"))

    # calculate anova
    print("\n")
    for model in df["model"].unique():
        temp_df = df[df["model"] == model]
        print(model + "one-way ANOVA result:")
        oneway_result = f_oneway(
            temp_df[temp_df["coh"] == "H"]["rt"],
            temp_df[temp_df["coh"] == "M"]["rt"],
            temp_df[temp_df["coh"] == "L"]["rt"],
            temp_df[temp_df["coh"] == "Z"]["rt"],
        )
        print(oneway_result)

    # Performing two-way ANOVA
    # for m in ["Shuffled feedback", "No feedback"]:
    #     temp_df = df[(df["model"] == "Full model") | (df["model"] == m)]
    #     model = ols(
    #         "rt ~ C(model) + C(coh) +\
    #     C(model):C(coh)",
    #         data=temp_df[temp_df["coh"] != "Z"],
    #     ).fit()
    #     result = sm.stats.anova_lm(model, type=2)
    #     print("\n")
    #     print("Two-way ANOVA compare %s Results:" % m)
    #     print(result)
    for m in ["Cut Specific", "Cut Nonspecific"]:
        temp_df = df[(df["model"] == "Full model") | (df["model"] == m)]
        model = ols(
            "rt ~ C(model) + C(coh) +\
        C(model):C(coh)",
            data=temp_df[temp_df["coh"] != "Z"],
        ).fit()
        result = sm.stats.anova_lm(model, type=2)
        print("\n")
        print("Two-way ANOVA compare %s Results:" % m)
        print(result)
    # m = "Remove feedback"
    # temp_df = df[(df["model"] == "Full model") | (df["model"] == m)]
    # model = ols(
    #     "rt ~ C(model) + C(coh) +\
    # C(model):C(coh)",
    #     data=temp_df[temp_df["coh"] != "Z"],
    # ).fit()
    # result = sm.stats.anova_lm(model, type=2)
    # print("\n")
    # print("Two-way ANOVA compare %s Results:" % m)
    # print(result)

    f.close()


def load_rt():
    if rerun_calculation:
        for f_dir in f_dirs:
            if "shuf" in f_dir:
                all_rt_df = pd.DataFrame(columns=["rep", "shuf", "coh", "rt", "model"])
                pbar = tqdm(total=total_rep * total_shuf)
                for rep in range(total_rep):
                    for shuf in range(total_shuf):
                        rt_temp = convert_rt2df(
                            calc_coh_rt(f_dir, rep, shuf), rep, shuf
                        )
                        rt_temp["model"] = ["Shuffled feedback"] * len(rt_temp.index)
                        all_rt_df = pd.concat([all_rt_df, rt_temp])
                        pbar.update(1)
                pbar.close()
                all_rt_df.to_csv(os.path.join(f_dir, "shufFeedback_all_rt.csv"))
            else:
                all_rt_df = pd.DataFrame(columns=["rep", "shuf", "coh", "rt", "model"])
                pbar = tqdm(total=total_rep)
                for rep in range(total_rep):
                    rt_temp = convert_rt2df(calc_coh_rt(f_dir, rep), rep)
                    if "noFeedback" in f_dir:
                        rt_temp["model"] = ["No feedback"] * len(rt_temp.index)
                    elif "removeFB" in f_dir:
                        rt_temp["model"] = ["Remove feedback"] * len(rt_temp.index)
                    elif "cutSpec" in f_dir:
                        rt_temp["model"] = ["Cut Specific"] * len(rt_temp.index)
                    elif "cutNonspec" in f_dir:
                        rt_temp["model"] = ["Cut Nonspecific"] * len(rt_temp.index)
                    else:
                        rt_temp["model"] = ["Full model"] * len(rt_temp.index)
                    all_rt_df = pd.concat([all_rt_df, rt_temp])
                    pbar.update(1)
                pbar.close()
                if "noFeedback" in f_dir:
                    all_rt_df.to_csv(os.path.join(f_dir, "noFeedback_all_rt.csv"))
                elif "removeFB" in f_dir:
                    all_rt_df.to_csv(os.path.join(f_dir, "removeFeedback_all_rt.csv"))
                elif "cutSpec" in f_dir:
                    all_rt_df.to_csv(os.path.join(f_dir, "cutSpec_all_rt.csv"))
                elif "cutNonspec" in f_dir:
                    all_rt_df.to_csv(os.path.join(f_dir, "cutNonspec_all_rt.csv"))
                else:
                    all_rt_df.to_csv(os.path.join(f_dir, "fullModel_all_rt.csv"))

    # shuf_df = pd.read_csv(os.path.join(f_dirs[2], "shufFeedback_all_rt.csv"))
    # nofb_df = pd.read_csv(os.path.join(f_dirs[1], "noFeedback_all_rt.csv"))
    full_df = pd.read_csv(os.path.join(f_dirs[0], "fullModel_all_rt.csv"))
    cut_spec_df = pd.read_csv(os.path.join(f_dirs[1], "cutSpec_all_rt.csv"))
    cut_nonspec_df = pd.read_csv(os.path.join(f_dirs[2], "cutNonspec_all_rt.csv"))
    # rmvfb_df = pd.read_csv(os.path.join(f_dirs[1], "removeFeedback_all_rt.csv"))

    # all_rt_df = combine_rt_dfs(full_df, nofb_df, shuf_df)
    # all_rt_df = combine_rt_dfs(full_df, rmvfb_df)
    all_rt_df = combine_rt_dfs(full_df, cut_spec_df, cut_nonspec_df)
    # below_thresh_count =  calc_below_thresh_trials(full_df, nofb_df, shuf_df)
    # below_thresh_count =  calc_below_thresh_trials(full_df, rmvfb_df)
    # below_thresh_count.to_csv(os.path.join(plt_dir, 'below_thresh_count.csv'))
    below_thresh_count = None
    return all_rt_df, below_thresh_count


def combine_rt_dfs(*args):
    df_list = []
    for df in args:
        temp_df = (
            df[["rep", "coh", "rt", "model"]].groupby(["model", "rep", "coh"]).mean()
        )
        df_list.append(temp_df)
    return pd.concat(df_list).reset_index()


def calc_below_thresh_trials(full_df, nofb_df, shuf_df):
    agg_func = {"rt": [lambda x: sum(x > 500)]}
    temp_full_df = (
        full_df[["rep", "coh", "rt", "model"]]
        .groupby(["model", "rep", "coh"])
        .agg(agg_func)
    )
    temp_nofb_df = (
        nofb_df[["rep", "coh", "rt", "model"]]
        .groupby(["model", "rep", "coh"])
        .agg(agg_func)
    )
    temp_shuf_df = (
        shuf_df[["rep", "shuf", "coh", "rt", "model"]]
        .groupby(["model", "rep", "shuf", "coh"])
        .agg(agg_func)
    )
    temp_shuf_df = temp_shuf_df.groupby(["model", "rep", "coh"]).mean()
    return pd.concat([temp_full_df, temp_nofb_df, temp_shuf_df]).reset_index()


def convert_rt2df(rt_dict, rep, shuf=np.nan):
    coh_arr = []
    all_rt_arr = np.array([])
    for coh in ["H", "M", "L", "Z"]:
        coh_arr.append([coh] * len(rt_dict[coh]))
        all_rt_arr = np.concatenate((all_rt_arr, rt_dict[coh]))
    rep_arr = [rep] * len(all_rt_arr)
    shuf_arr = [shuf] * len(all_rt_arr)
    coh_arr = sum(coh_arr, [])
    return pd.DataFrame(
        {"rep": rep_arr, "shuf": shuf_arr, "coh": coh_arr, "rt": all_rt_arr}
    )


main()
