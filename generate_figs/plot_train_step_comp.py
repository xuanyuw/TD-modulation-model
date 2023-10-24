import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from os.path import join
import tables
from utils import get_max_iter
from statannotations.Annotator import Annotator

# plot settings

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2
# plt.rcParams['figure.figsize'] = [6, 4]


f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    "trained_eqNum_removeFB_model",
]
model_types = ["Full model", "Remove feedback"]
total_rep = 50

# plt_dir = os.path.join('generate_figs', 'Fig7', '7c_ablation_acc_comp')
plt_dir = os.path.join("generate_figs", "rmv_fb_plots", "train_step_comp_eq_conn_num")
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)


df = pd.DataFrame(columns=["model", "max_iter"])
for i, f_dir in enumerate(f_dirs):
    iter_arr = []
    for rep in range(total_rep):
        train_table = tables.open_file(
            os.path.join(
                "crossOutput_noInterneuron_noMTConn_removeFB_model",
                "train_output_lr0.020000_rep%d.h5" % rep,
            ),
            mode="r",
        )
        train_table = train_table.root
        iter_arr.append(get_max_iter(train_table))
    model_arr = [model_types[i]] * total_rep
    df = pd.concat([df, pd.DataFrame({"model": model_arr, "max_iter": iter_arr})])

df["max_iter"] = df["max_iter"].astype(int)
colors = ["#FF0000", "#0080FE"]
fig, ax = plt.subplots()
sns.stripplot(x="model", y="max_iter", data=df, dodge=True, palette=colors, alpha=0.8)
# plot the mean line
sns.boxplot(
    showmeans=True,
    meanline=True,
    meanprops={"color": "gray", "ls": "--", "lw": 2},
    width=0.3,
    medianprops={"visible": False},
    whiskerprops={"visible": False},
    zorder=10,
    x="model",
    y="max_iter",
    data=df,
    showfliers=False,
    showbox=False,
    showcaps=False,
    ax=ax,
)

pairs = [("Full model", "Remove feedback")]

f = open(os.path.join(plt_dir, "stat_test.txt"), "w")
sys.stdout = f

annot = Annotator(ax, pairs, data=df, x="model", y="max_iter")
annot.configure(test="t-test_ind", text_format="star", loc="outside")
annot.apply_and_annotate()

plt.tight_layout()
ax.set(xlabel="", ylabel="# Iterations")

plt.savefig(join(plt_dir, "train_step_comparison.png"), format="png")
plt.savefig(join(plt_dir, "train_step_comparison.pdf"), format="pdf")
plt.savefig(join(plt_dir, "train_step_comparison.eps"), format="eps")
plt.close()
