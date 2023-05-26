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
        angle_dict = {}
        for f_dir in f_dirs:
            model_type = f_dir.split("_")[-2]
            angle_list = []
            for rep in tqdm(range(total_rep)):
                gl_points, rl_points = load_data(
                    os.path.join("slow_point_2D_plots", model_type, "rep%d" % rep)
                )
                ang = calculate_angle(gl_points, rl_points)
                angle_list.append(ang)
            angle_dict[model_type] = angle_list
        np.save(os.path.join(plot_dir, "angle_dict.npy"), angle_dict)
    else:
        angle_dict = np.load(
            os.path.join(plot_dir, "angle_dict.npy"), allow_pickle=True
        ).item()
    plot_angles(angle_dict)


def load_data(save_dir):
    with open(os.path.join(save_dir, "h_fp_transformed.pth"), "rb") as f:
        finder_dict = np.load(f, allow_pickle=True).item()
    gl_points = finder_dict["stim_gl"]
    rl_points = finder_dict["stim_rl"]
    return gl_points, rl_points


def calculate_angle(points1, points2):
    # Split the points into x, y, and z arrays for each group
    x1, y1, z1 = points1[:, 0], points1[:, 1], points1[:, 2]
    x2, y2, z2 = points2[:, 0], points2[:, 1], points2[:, 2]

    # Create the design matrices A1 and A2
    A1 = np.column_stack((x1, y1, np.ones_like(x1)))
    A2 = np.column_stack((x2, y2, np.ones_like(x2)))

    # Solve the least-squares problems to get the coefficients
    coeffs1, _, _, _ = np.linalg.lstsq(A1, z1, rcond=None)
    coeffs2, _, _, _ = np.linalg.lstsq(A2, z2, rcond=None)

    # Calculate the direction vectors of the lines
    dir_vec1 = np.array([coeffs1[0], coeffs1[1], -1])
    dir_vec2 = np.array([coeffs2[0], coeffs2[1], -1])

    # Calculate the angle between the lines using the dot product
    cos_angle = np.dot(dir_vec1, dir_vec2) / (
        np.linalg.norm(dir_vec1) * np.linalg.norm(dir_vec2)
    )
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def plot_angles(angle_dict):
    # plot a violin plot of the angles

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    df = pd.DataFrame.from_dict(angle_dict)
    df = df.melt(var_name="Model", value_name="Angle")

    sns.violinplot(
        data=df,
        x="Model",
        y="Angle",
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
        y="Angle",
    )
    annot.configure(test="t-test_paired", text_format="star", loc="outside")
    annot.apply_and_annotate()

    ax.set_xticklabels(list(angle_dict.keys()))
    ax.set_ylabel("Angle (degree)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "angle_violin.png"))
    plt.close()


if __name__ == "__main__":
    main()
