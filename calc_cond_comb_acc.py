import os
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from utils import load_test_data, min_max_normalize, recover_targ_loc
from scipy.io import savemat
from utils import combine_idx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

lr = 0.02
total_rep = 50
correct_only = True
rerun_calc = False
save_plot = True

f_dirs = [
    "test_output_full_model",
    "test_output_noFeedback_model",
    "cutSpec_model",
    "cutNonspec_model",
]

data_dir = "dPCA_allTrial_data"


def main():
    for f_dir in f_dirs:
        model_type = f_dir.split("_")[-2]
        # sepStim_dir = os.path.join(data_dir, model_type, "sepStim")
        # if not os.path.exists(sepStim_dir):
        #     os.makedirs(sepStim_dir)
        # sepSac_dir = os.path.join(data_dir, model_type, "sepSac")
        # if not os.path.exists(sepSac_dir):
        #     os.makedirs(sepSac_dir)

        # save_fn_sepStim = os.path.join(
        #     data_dir, model_type, "sepStim", "all_sepStim_acc.csv"
        # )
        # save_fn_sepSac = os.path.join(
        #     data_dir, model_type, "sepSac", "all_sepSac_acc.csv"
        # )
        allTrials_dir = os.path.join(data_dir, model_type, "allTrials")
        save_fn_allTrials = os.path.join(allTrials_dir, "allTrials_acc.csv")
        if (
            rerun_calc
            or not os.path.exists(save_fn_allTrials)
            # or not os.path.exists(save_fn_sepStim)
            # or not os.path.exists(save_fn_sepSac)
        ):
            # all_stim_df = pd.DataFrame()
            # all_sac_df = pd.DataFrame()
            all_allTrials_df = pd.DataFrame()
            for rep in tqdm(range(total_rep)):
                n = SimpleNamespace(
                    **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
                )
                allTrials_df = get_trialCnt_df_allTrials(n, rep)
                all_allTrials_df = pd.concat([all_allTrials_df, allTrials_df])
            all_allTrials_df.to_csv(save_fn_allTrials)
            # stim_df, sac_df = get_acc_df(n, rep)
            # stim_df.to_csv(save_fn_sepStim)
            # sac_df.to_csv(save_fn_sepSac)
            #     all_stim_df = pd.concat([all_stim_df, stim_df])
            #     all_sac_df = pd.concat([all_sac_df, sac_df])
            # all_stim_df.to_csv(save_fn_sepStim)
            # all_sac_df.to_csv(save_fn_sepSac)
        # else:
        # all_stim_df = pd.read_csv(save_fn_sepStim)
        # all_sac_df = pd.read_csv(save_fn_sepSac)
        # plot_acc_and_count(
        #     all_stim_df, "Stim_dir_targ_arr_combinations", sepStim_dir, save_plot
        # )
        # plot_acc_and_count(
        #     all_stim_df, "Sac_dir_targ_arr_combinations", sepSac_dir, save_plot
        # )
        # plot_acc_and_count_summary(
        #     all_stim_df,
        #     "%s_Stim_dir_targ_arr_combinations_summary" % model_type,
        #     sepStim_dir,
        #     save_plot,
        # )
        # plot_acc_and_count_summary(
        #     all_sac_df,
        #     "%s_Sac_dir_targ_arr_combinations_summary" % model_type,
        #     sepSac_dir,
        #     save_plot,
        # )


def plot_acc_and_count_summary(df, title, save_dir, save_plot):
    # plot boxplots for each condition
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.stripplot(
        x="label",
        y="acc",
        data=df,
        ax=ax[0],
    ).set(title="Accuracy", ylim=(0, 1.1))
    sns.stripplot(
        x="label",
        y="total",
        data=df,
        ax=ax[1],
    ).set(title="Count", ylim=(0, 1100))
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(save_dir, title + ".png"))


def get_trialCnt_df_allTrials(n, rep):
    stim_dir = n.stim_dir
    targ_arrange = recover_targ_loc(n.desired_out, n.stim_dir)[-1, :]

    cond1_total = np.sum(combine_idx(n.choice == 0, stim_dir == 135, targ_arrange == 0))
    cond2_total = np.sum(combine_idx(n.choice == 0, stim_dir == 135, targ_arrange == 1))
    cond3_total = np.sum(combine_idx(n.choice == 0, stim_dir == 315, targ_arrange == 0))
    cond4_total = np.sum(combine_idx(n.choice == 0, stim_dir == 315, targ_arrange == 1))
    cond5_total = np.sum(combine_idx(n.choice == 1, stim_dir == 135, targ_arrange == 0))
    cond6_total = np.sum(combine_idx(n.choice == 1, stim_dir == 135, targ_arrange == 1))
    cond7_total = np.sum(combine_idx(n.choice == 1, stim_dir == 315, targ_arrange == 0))
    cond8_total = np.sum(combine_idx(n.choice == 1, stim_dir == 315, targ_arrange == 1))

    stim_label_arr = np.array([135, 135, 315, 315] * 2)
    sac_label_arr = np.array(["left"] * 4 + ["right"] * 4)
    targ_arrange_arr = np.array(["greenL", "redL"] * 4)

    allTrials_df = pd.DataFrame(
        {
            "rep": np.array([rep] * len(stim_label_arr)),
            "stim": stim_label_arr,
            "targ_arrange": targ_arrange_arr,
            "sac": sac_label_arr,
            "total": np.array(
                [
                    cond1_total,
                    cond2_total,
                    cond3_total,
                    cond4_total,
                    cond5_total,
                    cond6_total,
                    cond7_total,
                    cond8_total,
                ]
            ),
        }
    )

    return allTrials_df


def get_acc_df(n, rep):
    stim_dir = n.stim_dir
    targ_arrange = recover_targ_loc(n.desired_out, n.stim_dir)[-1, :]

    if correct_only:
        cond1_total = sum(combine_idx(stim_dir == 135, targ_arrange == 0))
        cond1_acc = (
            sum(combine_idx(stim_dir == 135, targ_arrange == 0, n.correct_idx))
            / cond1_total
        )
        cond2_total = sum(combine_idx(stim_dir == 135, targ_arrange == 1))
        cond2_acc = (
            sum(combine_idx(stim_dir == 135, targ_arrange == 1, n.correct_idx))
            / cond2_total
        )
        cond3_total = sum(combine_idx(stim_dir == 315, targ_arrange == 0))
        cond3_acc = (
            sum(combine_idx(stim_dir == 315, targ_arrange == 0, n.correct_idx))
            / cond3_total
        )
        cond4_total = sum(combine_idx(stim_dir == 315, targ_arrange == 1))
        cond4_acc = (
            sum(combine_idx(stim_dir == 315, targ_arrange == 1, n.correct_idx))
            / cond4_total
        )
        cond5_total = sum(combine_idx(n.choice == 0, targ_arrange == 0))
        cond5_acc = (
            sum(combine_idx(n.choice == 0, targ_arrange == 0, n.correct_idx))
            / cond5_total
        )
        cond6_total = sum(combine_idx(n.choice == 0, targ_arrange == 1))
        cond6_acc = (
            sum(combine_idx(n.choice == 0, targ_arrange == 1, n.correct_idx))
            / cond6_total
        )
        cond7_total = sum(combine_idx(n.choice == 1, targ_arrange == 0))
        cond7_acc = (
            sum(combine_idx(n.choice == 1, targ_arrange == 0, n.correct_idx))
            / cond7_total
        )
        cond8_total = sum(combine_idx(n.choice == 1, targ_arrange == 1))
        cond8_acc = (
            sum(combine_idx(n.choice == 1, targ_arrange == 1, n.correct_idx))
            / cond8_total
        )

        stim_label_arr = ["135_greenL", "135_redL", "315_greenL", "315_redL"]
        sac_label_arr = ["left_greenL", "left_redL", "right_greenL", "right_redL"]

        stim_df = pd.DataFrame(
            {
                "rep": np.array([rep] * len(stim_label_arr)),
                "label": stim_label_arr,
                "total": np.array([cond1_total, cond2_total, cond3_total, cond4_total]),
                "acc": np.array([cond1_acc, cond2_acc, cond3_acc, cond4_acc]),
            }
        )
        sac_df = pd.DataFrame(
            {
                "rep": np.array([rep] * len(sac_label_arr)),
                "label": sac_label_arr,
                "total": np.array([cond5_total, cond6_total, cond7_total, cond8_total]),
                "acc": np.array([cond5_acc, cond6_acc, cond7_acc, cond8_acc]),
            }
        )

    return stim_df, sac_df


def plot_acc_and_count(df, title, plt_dir, save_plot):
    # plot accuracy for each condition
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.barplot(x="rep", y="acc", hue="label", data=df, ax=ax).set(ylim=(0, 1.1))
    plt.legend(loc="best")

    plt.suptitle("%s Accuracy" % title)
    plt.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(plt_dir, title + "_acc.png"))

    # plot total count for each condition
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.barplot(x="rep", y="total", hue="label", data=df, ax=ax).set(ylim=(0, 1100))
    plt.legend(loc="best")
    plt.suptitle("%s Trial Count" % title)
    plt.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(plt_dir, title + "_count.png"))


if __name__ == "__main__":
    main()
