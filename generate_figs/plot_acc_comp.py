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
from statannotations.Annotator import Annotator
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import to_rgb
from tqdm import tqdm
from statsmodels.formula.api import ols
import statsmodels.api as sm

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
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_shufFeedback_model"
# ]
# f_dirs = [
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
#     "cutSpec_model",
#     "cutNonspec_model",
# ]
# f_dirs = [
#     "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
#     "trained_removeFB_model",
# ]
f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    "crossOutput_noInterneuron_noMTConn_removeFB_model",
]

# plt_dir = os.path.join('generate_figs', 'Fig7', '7c_ablation_acc_comp')
plt_dir = os.path.join("generate_figs", "rmv_fb_plots", "acc_comp")
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)


total_rep = 50
total_shuf = 100
lr = 2e-2
plot_sel = True
rerun_calculation = True
plot_trained = True
save_plot = False


def main():
    df = load_acc()
    fig, ax = plt.subplots()
    colors = ["#FF0000", "#00FF00", "#0000FF", "#424242"]
    handles = []
    # sns.violinplot(
    #     x="coh",
    #     y="acc",
    #     hue="model",
    #     data=df,
    #     inner="points",
    #     ax=ax,
    #     palette=[".2", ".5"],
    #     hue_order=["Full model", "No feedback", "Shuffled feedback"],
    # )
    sns.violinplot(
        x="coh",
        y="acc",
        hue="model",
        data=df,
        inner="points",
        ax=ax,
        palette=[".2", ".5"],
        hue_order=["Full model", "Remove feedback"],
    )

    # sns.violinplot(
    #     x="coh",
    #     y="acc",
    #     hue="model",
    #     data=df,
    #     inner="points",
    #     ax=ax,
    #     palette=[".2", ".5"],
    #     hue_order=["Full model", "Cut Nonspecific", "Cut Specific"],
    # )
    # for ind, violin in enumerate(ax.findobj(PolyCollection)):
    #     rgb = to_rgb(colors[ind // 3])
    #     if ind % 3 == 1:
    #         rgb = 0.4 + 0.6 * np.array(rgb)  # make whiter
    #     if ind % 3 == 2:
    #         rgb = 0.7 + 0.3 * np.array(rgb)  # make whiter
    #     violin.set_facecolor(rgb)
    #     handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor="black"))

    for ind, violin in enumerate(ax.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 == 1:
            rgb = 0.4 + 0.6 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor="black"))

    # ax.legend(
    #     handles=[tuple(handles[::3]), tuple(handles[1::3]), tuple(handles[2::3])],
    #     labels=["Full model", "Cut Nonspecific", "Cut Specific"],
    #     # labels=df["model"].astype("category").cat.categories.to_list(),
    #     handlelength=4,
    #     handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
    #     loc="lower left",
    #     frameon=False,
    # )
    ax.legend(
        handles=[tuple(handles[::2]), tuple(handles[1::2])],
        labels=df["model"].astype("category").cat.categories.to_list(),
        handlelength=4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="lower left",
        frameon=False,
    )

    # add statistical test results
    # pairs = [(('H', 'Full model'), ('H', 'No feedback')), (('H', 'Full model'), ('H', 'Shuffled feedback')),
    #             (('M', 'Full model'), ('M', 'No feedback')), (('M', 'Full model'), ('M', 'Shuffled feedback')),
    #             (('L', 'Full model'), ('L', 'No feedback')), (('L', 'Full model'), ('L', 'Shuffled feedback')),
    #             (('Z', 'Full model'), ('Z', 'No feedback')), (('Z', 'Full model'), ('Z', 'Shuffled feedback'))]

    pairs = [
        (("H", "Full model"), ("H", "Remove feedback")),
        (("M", "Full model"), ("M", "Remove feedback")),
        (("L", "Full model"), ("L", "Remove feedback")),
        (("Z", "Full model"), ("Z", "Remove feedback")),
    ]
    # pairs = [
    #     (("H", "Full model"), ("H", "Cut Specific")),
    #     (("H", "Full model"), ("H", "Cut Nonspecific")),
    #     (("M", "Full model"), ("M", "Cut Specific")),
    #     (("M", "Full model"), ("M", "Cut Nonspecific")),
    #     (("L", "Full model"), ("L", "Cut Specific")),
    #     (("L", "Full model"), ("L", "Cut Nonspecific")),
    #     (("Z", "Full model"), ("Z", "Cut Specific")),
    #     (("Z", "Full model"), ("Z", "Cut Nonspecific")),
    # ]

    f = open(os.path.join(plt_dir, "stat_test.txt"), "w")
    sys.stdout = f

    # annot = Annotator(
    #     ax,
    #     pairs,
    #     data=df,
    #     x="coh",
    #     y="acc",
    #     hue="model",
    #     order=["H", "M", "L", "Z"],
    #     hue_order=["Full model", "Cut Nonspecific", "Cut Specific"],
    # )
    annot = Annotator(
        ax,
        pairs,
        data=df,
        x="coh",
        y="acc",
        hue="model",
        order=["H", "M", "L", "Z"],
        hue_order=["Full model", "Remove feedback"],
    )
    annot.configure(test="t-test_ind", text_format="star", loc="outside")
    annot.apply_and_annotate()

    ax.set(xlabel="Coherence", ylabel="Accuracy")
    ax.tick_params(bottom=True, left=True)
    plt.tight_layout()

    # plt.show()
    if save_plot:
        plt.savefig(os.path.join(plt_dir, "ablation_acc_comp.pdf"), format="pdf")
        plt.savefig(os.path.join(plt_dir, "ablation_acc_comp.eps"), format="eps")
        plt.savefig(os.path.join(plt_dir, "ablation_acc_comp.png"), format="png")

    df_mean = df[["model", "coh", "acc"]].groupby(["model", "coh"]).mean()
    df_mean.to_csv(os.path.join(plt_dir, "all_model_coh_acc.csv"))

    # Performing two-way ANOVA
    # for m in ["Cut Specific", "Cut Nonspecific"]:
    #     temp_df = df[(df["model"] == "Full model") | (df["model"] == m)]
    #     model = ols(
    #         "acc ~ C(model) + C(coh) +\
    #     C(model):C(coh)",
    #         data=temp_df[temp_df["coh"] != "Z"],
    #     ).fit()
    #     result = sm.stats.anova_lm(model, type=2)
    #     print("\n")
    #     print("Two-way ANOVA compare %s Results:" % m)
    #     print(result)
    # for m in ['Shuffled feedback', 'No feedback']:
    #     temp_df = df[(df['model']=='Full model')| (df['model']==m)]
    #     model = ols('acc ~ C(model) + C(coh) +\
    #     C(model):C(coh)',
    #                 data=temp_df[temp_df['coh']!='Z']).fit()
    #     result = sm.stats.anova_lm(model, type=2)
    #     print('\n')
    #     print('Two-way ANOVA compare %s Results:'%m)
    #     print(result)
    m = "Remove feedback"
    temp_df = df[(df["model"] == "Full model") | (df["model"] == m)]
    model = ols(
        "acc ~ C(model) + C(coh) +\
    C(model):C(coh)",
        data=temp_df[temp_df["coh"] != "Z"],
    ).fit()
    result = sm.stats.anova_lm(model, type=2)
    print("\n")
    print("Two-way ANOVA compare %s Results:" % m)
    print(result)

    f.close()


def load_acc():
    if rerun_calculation:
        for f_dir in f_dirs:
            if "shuf" in f_dir:
                all_acc_df = pd.DataFrame(columns=["rep", "shuf", "coh", "acc"])
                pbar = tqdm(total=total_rep)
                for rep in range(total_rep):
                    for shuf in range(total_shuf):
                        with open(
                            os.path.join(
                                f_dir, "test_results_%d_shuf%d.pkl" % (rep, shuf)
                            ),
                            "rb",
                        ) as f:
                            data = load(f)
                        acc_df = pd.DataFrame(
                            {
                                "rep": [rep] * 4,
                                "shuf": [shuf] * 4,
                                "coh": ["H", "M", "L", "Z"],
                                "acc": [
                                    data["H_acc"][0],
                                    data["M_acc"][0],
                                    data["L_acc"][0],
                                    data["Z_acc"][0],
                                ],
                            }
                        )
                        all_acc_df = pd.concat([all_acc_df, acc_df], ignore_index=True)
                    pbar.update(1)
                all_acc_df.to_csv(os.path.join(f_dir, "all_test_acc.csv"))
                shuf_acc_df = all_acc_df
            else:
                all_acc_df = pd.DataFrame(columns=["rep", "coh", "acc"])
                for rep in range(total_rep):
                    if "cut" in f_dir:
                        with open(
                            os.path.join(f_dir, "test_results_%d_cut.pkl" % (rep)), "rb"
                        ) as f:
                            data = load(f)
                    else:
                        with open(
                            os.path.join(f_dir, "test_results_%d.pkl" % (rep)), "rb"
                        ) as f:
                            data = load(f)
                    acc_df = pd.DataFrame(
                        {
                            "rep": [rep] * 4,
                            "coh": ["H", "M", "L", "Z"],
                            "acc": [
                                data["H_acc"][0],
                                data["M_acc"][0],
                                data["L_acc"][0],
                                data["Z_acc"][0],
                            ],
                        }
                    )
                    all_acc_df = pd.concat([all_acc_df, acc_df], ignore_index=True)
                all_acc_df.to_csv(os.path.join(f_dir, "all_test_acc.csv"))
                if "noFeedback" in f_dir:
                    nofb_acc_df = all_acc_df
                elif "removeFB" in f_dir:
                    rmvfb_acc_df = all_acc_df
                elif "cutSpec" in f_dir:
                    cut_spec_acc_df = all_acc_df
                elif "cutNonspec" in f_dir:
                    cut_nonspec_acc_df = all_acc_df

                else:
                    full_acc_df = all_acc_df
    else:
        full_acc_df = pd.read_csv(os.path.join(f_dirs[0], "all_test_acc.csv"))
        # cut_spec_acc_df = pd.read_csv(os.path.join(f_dirs[1], "all_test_acc.csv"))
        # cut_nonspec_acc_df = pd.read_csv(os.path.join(f_dirs[2], "all_test_acc.csv"))
        rmvfb_acc_df = pd.read_csv(os.path.join(f_dirs[1], "all_test_acc.csv"))
        # nofb_acc_df = pd.read_csv(os.path.join(f_dirs[1], 'all_test_acc.csv'))
        # shuf_acc_df = pd.read_csv(os.path.join(f_dirs[2], 'all_test_acc.csv'))

    # combine all acc df
    # shuf_df_temp = shuf_acc_df[['rep', 'coh', 'acc']].groupby(['rep', 'coh']).mean().reset_index()
    full_acc_df["model"] = ["Full model"] * len(full_acc_df.index)
    # cut_spec_acc_df["model"] = ["Cut Specific"] * len(cut_spec_acc_df.index)
    # cut_nonspec_acc_df["model"] = ["Cut Nonspecific"] * len(cut_nonspec_acc_df.index)
    rmvfb_acc_df["model"] = ["Remove feedback"] * len(rmvfb_acc_df.index)

    # nofb_acc_df['model'] = ['No feedback']*len(nofb_acc_df.index)
    # shuf_df_temp['model'] = ['Shuffled feedback']*len(shuf_df_temp.index)

    # return pd.concat([full_acc_df, nofb_acc_df, shuf_df_temp], ignore_index=True)
    return pd.concat([full_acc_df, rmvfb_acc_df], ignore_index=True)
    # return pd.concat(
    #     [full_acc_df, cut_spec_acc_df, cut_nonspec_acc_df], ignore_index=True
    # )


main()
