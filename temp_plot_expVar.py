import os
import sys
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator

lr = 0.02
total_rep = 50
useCorrect = False

# model_types = [
#     "full",
#     "noFeedback",
#     "cutSpec",
#     "cutNonspec",
# ]

model_types = [
    "full",
    "removeFB",
]

data_dir = (
    "F:\\Github\\TD-modulation-model\\dPCA_allTrial_plots\\allTrials_wInter\\MotionOnly"
)


def main():
    explVar_dict_condInd = {}
    # explVar_dict_dir = {}
    explVar_dict_targArr = {}
    # explVar_dict_inter = {}
    explVar_dict_sac = {}
    explVar_dict_stim = {}
    for mType in model_types:
        condInd_arr = []
        # dir_arr = []
        targArr_arr = []
        # inter_arr = []
        sac_arr = []
        stim_arr = []
        for rep in tqdm(range(total_rep)):
            # if useCorrect:
            #     data = loadmat(
            #         os.path.join(
            #             data_dir,
            #             mType,
            #             "rep%d_dPCA_expVar_correctOnly.mat" % (rep + 1),
            #         )
            #     )
            # else:
            if not os.path.exists(
                os.path.join(
                    data_dir, mType, "rep%d_dPCA_expVar_allTrials.mat" % (rep + 1)
                )
            ):
                sac_arr.append(np.nan)
                stim_arr.append(np.nan)
                targArr_arr.append(np.nan)
                condInd_arr.append(np.nan)
                continue

            data = loadmat(
                os.path.join(
                    data_dir, mType, "rep%d_dPCA_expVar_allTrials.mat" % (rep + 1)
                )
            )
            explVar = data["explVar"]["margVar"][0][0]
            explVarSum = np.sum(explVar, axis=1)
            # explVarSum = (explVarSum / np.sum(explVarSum)) * 100
            # explVarSum = (
            #     data["explVar"]["totalMarginalizedVar"][0][0]
            #     / data["explVar"]["totalVar"][0][0]
            # )[0]*100

            sac_arr.append(explVarSum[0])
            stim_arr.append(explVarSum[1])
            targArr_arr.append(explVarSum[2])
            condInd_arr.append(explVarSum[3])
            # dir_arr.append(explVarSum[0])
            # targArr_arr.append(explVarSum[1])
            # condInd_arr.append(explVarSum[2])
            # inter_arr.append(explVarSum[3])

        # explVar_dict_dir[mType] = dir_arr
        explVar_dict_sac[mType] = sac_arr
        explVar_dict_targArr[mType] = targArr_arr
        explVar_dict_condInd[mType] = condInd_arr
        explVar_dict_stim[mType] = stim_arr

    f = open(os.path.join(data_dir, "stat_test.txt"), "w")
    sys.stdout = f

    plot_explVars(explVar_dict_stim, "Stimulus Explained Variance")
    plot_explVars(explVar_dict_sac, "Saccade Explained Variance")
    plot_explVars(explVar_dict_targArr, "Target Arrangement Explained Variance")
    plot_explVars(explVar_dict_condInd, "Condition-Independent Explained Variance")

    # if useCorrect:
    #     plot_explVars(
    #         explVar_dict_condInd, "Condition-Independent Explained Variance CorrectOnly"
    #     )
    # else:
    #     plot_explVars(explVar_dict_condInd, "Condition-Independent Explained Variance")
    # if "Stim" in data_dir:
    #     if useCorrect:
    #         plot_explVars(
    #             explVar_dict_dir, "Stimulus Direction Explained Variance CorrectOnly"
    #         )
    #     else:
    #         plot_explVars(explVar_dict_dir, "Stimulus Direction Explained Variance")
    # elif "Sac" in data_dir:
    #     plot_explVars(explVar_dict_dir, "Saccade Direction Explained Variance")

    # if useCorrect:
    #     plot_explVars(
    #         explVar_dict_targArr, "Target Arrangement Explained Variance CorrectOnly"
    #     )
    #     plot_explVars(explVar_dict_inter, "Interaction Explained Variance CorrectOnly")
    # else:
    #     plot_explVars(explVar_dict_targArr, "Target Arrangement Explained Variance")
    #     plot_explVars(explVar_dict_inter, "Interaction Explained Variance")

    f.close()


def plot_explVars(explVar_dict, title):
    # plot a violin plot of the explVars

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    df = pd.DataFrame.from_dict(explVar_dict)
    df.dropna(inplace=True, how="any")

    # find the columns with nan value and fill with column mean
    # for col in df.columns:
    #     if df[col].isnull().values.any():
    #         df[col] = df[col].fillna(df[col].mean())
    df = df.melt(var_name="Model", value_name="explVar")

    sns.violinplot(
        data=df,
        x="Model",
        y="explVar",
        palette="Set3",
        ax=ax,
        cut=0,
        inner="points",
        # order=["full", "noFeedback", "cutSpec", "cutNonspec"],
    )

    # pairs = [
    #     ("full", "noFeedback"),
    #     ("full", "cutSpec"),
    #     ("full", "cutNonspec"),
    #     ("cutSpec", "cutNonspec"),
    # ]

    pairs = [
        ("full", "removeFB"),
    ]
    annot = Annotator(
        ax,
        pairs,
        data=df,
        x="Model",
        y="explVar",
        # order=["full", "noFeedback", "cutSpec", "cutNonspec"],
    )
    annot.configure(test="Wilcoxon", text_format="star")
    print("Statistical test results for %s:" % title)
    annot.apply_and_annotate()
    print("-----------------------")

    ax.set_xticklabels(list(explVar_dict.keys()))
    ax.set_ylabel("explVar %")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "%s.png" % title))
    plt.savefig(os.path.join(data_dir, "%s.pdf" % title))
    plt.close()


if __name__ == "__main__":
    main()
