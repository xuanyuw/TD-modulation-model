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
from os.path import join
import numpy as np
import tables
from utils import (
    find_pref_dir,
    find_pref_targ_color,
    pick_selective_neurons,
    min_max_normalize,
    recover_targ_loc,
    relu,
)
from model.calc_params import par
from statannotations.Annotator import Annotator
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple

# plot settings
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2


f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    "trained_eqNum_removeFB_model",
]

plt_dir = os.path.join("generate_figs", "rmv_fb_plots")


if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)


total_rep = range(50)
# total_rep = [2, 5, 8, 14, 16, 19, 25, 44]
# total_rep = np.array([x for x in range(50) if x not in [2, 5, 8, 14, 16, 19, 25, 44]])
total_shuf = 100
lr = 2e-2
plot_sel = True
rerun_calculation = False
plot_trained = True


def main():
    df = load_data()

    plot_w_distr(df)


def load_data():
    all_df = pd.DataFrame()
    for f_dir in f_dirs:
        model_type = f_dir.split("_")[-2]
        title = "%s_weight_comparison" % model_type
        df = pd.read_csv(join(f_dir, title + "_selective_trained_data.csv"))
        if model_type == "highTestCoh":
            model_type = "full"
        df["model_type"] = model_type
        all_df = all_df.append(df)
    return all_df


def plot_w_distr(df, rep=None):
    df = df[df["rep"].isin(total_rep)]
    df = df[df["conn_type"] == "ff"]

    popu_df = (
        df[["rep", "conn", "weights", "model_type"]]
        .groupby(["rep", "conn", "model_type"])
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots()
    colors = ["#FF0000", "#0080FE"]

    sns.barplot(
        x="conn", y="weights", hue="model_type", data=popu_df, ax=ax, palette=colors
    )
    # plt.setp(ax.collections, alpha=.9)
    ax.set(xlabel="", ylabel="Weight")
    plt.legend(frameon=False, loc="best")

    # plot significance values

    pairs = [
        (("mr-tr", "full"), ("mr-tr", "removeFB")),
        (("mr-tg", "full"), ("mr-tg", "removeFB")),
        (("mg-tr", "full"), ("mg-tr", "removeFB")),
        (("mg-tg", "full"), ("mg-tg", "removeFB")),
    ]
    annot = Annotator(ax, pairs, data=popu_df, x="conn", y="weights", hue="model_type")
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()
    plt.tight_layout()

    plt.savefig(
        join(plt_dir, "weight_comp_full_rmvFB.png"), format="png", bbox_inches="tight"
    )
    # plt.savefig(join(plt_dir, title + ".pdf"), format="pdf", bbox_inches="tight")
    # plt.savefig(join(plt_dir, title + ".eps"), format="eps", bbox_inches="tight")


if __name__ == "__main__":
    main()
