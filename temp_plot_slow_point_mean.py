import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator


lr = 0.02
# rep = 0
total_rep = 50
# plot_coh = "H"


f_dirs = [
    "test_output_full_model",
    "test_output_noFeedback_model",
    "cutSpec_model",
    "cutNonspec_model",
]
all_cohs = ["H", "M", "L", "Z"]
rerun_calculation = True

plot_dir = "slow_point_2D_plots"


def main():
    if rerun_calculation:
        Distances_dict = {}
        for f_dir in f_dirs:
            model_type = f_dir.split("_")[-2]
            Distances_list = []
            for rep in tqdm(range(total_rep)):
                gl_points, rl_points = load_data(
                    os.path.join("slow_point_2D_plots", model_type, "rep%d" % rep)
                )
                ang = calculate_mean_dist(gl_points, rl_points)
                Distances_list.append(ang)
            Distances_dict[model_type] = Distances_list
        np.save(os.path.join(plot_dir, "Distances_dict.npy"), Distances_dict)
    else:
        Distances_dict = np.load(
            os.path.join(plot_dir, "Distances_dict.npy"), allow_pickle=True
        ).item()
    plot_Distancess(Distances_dict)


def load_data(save_dir):
    with open(os.path.join(save_dir, "h_fp_transformed.pth"), "rb") as f:
        finder_dict = np.load(f, allow_pickle=True).item()
    gl_points = finder_dict["stim_gl"]
    rl_points = finder_dict["stim_rl"]
    return gl_points, rl_points


def calculate_mean_dist(points1, points2):
    # calculate the distance between the averages of two sets of points
    # Split the points into x, y, and z arrays for each group
    x1, y1, z1 = points1[:, 0], points1[:, 1], points1[:, 2]
    x2, y2, z2 = points2[:, 0], points2[:, 1], points2[:, 2]

    # Calculate the average of each group
    x1_avg, y1_avg, z1_avg = np.mean(x1), np.mean(y1), np.mean(z1)
    x2_avg, y2_avg, z2_avg = np.mean(x2), np.mean(y2), np.mean(z2)

    # Calculate the distance between the two averages
    dist = np.sqrt(
        (x1_avg - x2_avg) ** 2 + (y1_avg - y2_avg) ** 2 + (z1_avg - z2_avg) ** 2
    )
    return dist


def plot_Distancess(Distances_dict):
    # plot a violin plot of the Distancess

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    df = pd.DataFrame.from_dict(Distances_dict)
    df = df.melt(var_name="Model", value_name="Distances")

    sns.violinplot(
        data=df,
        x="Model",
        y="Distances",
        palette="Set3",
        ax=ax,
        cut=0,
    )

    pairs = [("full", "noFeedback"), ("full", "cutSpec"), ("full", "cutNonspec")]
    annot = Annotator(
        ax,
        pairs,
        data=df,
        x="Model",
        y="Distances",
    )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()

    ax.set_xticklabels(list(Distances_dict.keys()))
    ax.set_ylabel("Distance between mean points")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "mean_dist_violin.png"))
    plt.close()


if __name__ == "__main__":
    main()
