import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import *
from types import SimpleNamespace
from pickle import load, dump
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from scipy.stats import sem

from svm_cell_act import run_pca_all_model, run_SVM_all_model, load_sac_act

# plot settings
plt.rcParams["figure.figsize"] = [10, 4]
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
    "cutNonspec_model",
    "cutSpec_model",
]
pic_dir = os.path.join("new_dyn_analysis", "plots")
data_dir = os.path.join("new_dyn_analysis", "data")


def main():
    all_act_mat_pca = run_pca_all_model(f_dirs, False)
    all_model_acc_li, all_coef_li, all_intercept_li = run_SVM_all_model(
        f_dirs, all_act_mat_pca, rerun_calc=False
    )
    # plot_SVM_example(all_act_mat_pca, all_coef_li, all_intercept_li, 3)
    all_proj_li = project_all_data(
        all_act_mat_pca,
        all_coef_li,
        all_intercept_li,
        all_model_acc_li,
        rerun_calc=False,
    )
    # plot_proj_example(all_proj_li, 3)
    calc_all_potential(all_proj_li, rerun_calc=True)


def plot_proj_example(all_proj_li, rep):
    _, _, label = load_sac_act(f_dirs[0], rep, reload=False)
    example = all_proj_li[0][rep][:, -1]
    plt.scatter(example[label == 0], np.ones(sum(label == 0)), c="g")
    plt.scatter(example[label == 1], np.ones(sum(label == 1)) * 2, c="r")
    plt.show()


def plot_SVM_example(all_act_mat_pca, all_coef_li, all_intercept_li, rep):
    _, _, label = load_sac_act(f_dirs[0], rep, reload=False)
    plot_SVM(
        all_act_mat_pca[0][rep][:, :, -1],
        label,
        all_coef_li[0][-1][0, :, rep],
        all_intercept_li[0][-1, 0, rep],
    )
    # save plot
    # plt.savefig(os.path.join(pic_dir, "SVM_example.png"), dpi=300, bbox_inches="tight")


def plot_SVM(X, Y, coef, intercept):
    """
    plot the SVM hyperplane and the choice axis
    """
    # plot the SVM hyperplane

    z = lambda x, y: (-intercept - coef[0] * x - coef[1] * y) / coef[2]

    tmp = np.linspace(-10, 10, 30)
    x, y = np.meshgrid(tmp, tmp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X = X.T
    ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], "og", alpha=0.5)
    ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], "or", alpha=0.5)
    ax.plot_surface(x, y, z(x, y))

    # plot the choice axis
    orth_base = np.linspace(-10, 10, 100)
    ax.plot3D(
        orth_base * coef[0],
        orth_base * coef[1],
        orth_base * coef[2],
        color="black",
    )
    ax.view_init(8, -12)


def project_to_choice_axis(choice_axis, data):
    """
    project data to the choice axis
    """
    new_data = np.empty((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            new_data[i, j] = np.dot(data[:, i, j], choice_axis)
    return new_data


def project_all_data(
    all_act_mat_pca,
    all_coef_li,
    all_intercept_li,
    all_model_acc_li,
    total_rep=50,
    rerun_calc=False,
):
    """
    project all data to the choice axis
    """
    if rerun_calc:
        all_proj_li = []
        for coef, intercept, act_mat_pca, acc in zip(
            all_coef_li, all_intercept_li, all_act_mat_pca, all_model_acc_li
        ):
            model_proj_li = []
            for rep in tqdm(range(total_rep)):
                model_act_mat = act_mat_pca[rep]
                # use best performing of the 10 CV results as the choice axis
                best_model_idx = np.argmax(acc[-1, :, rep])
                # model_intercept = intercept[-1, best_model_idx, rep]
                choice_axis = coef[-1, best_model_idx, :, rep]
                proj = project_to_choice_axis(choice_axis, act_mat_pca[rep])
                model_proj_li.append(proj)
            all_proj_li.append(model_proj_li)
        # save data
        with open(os.path.join(data_dir, "all_proj_li.pkl"), "wb") as f:
            dump(all_proj_li, f)
    else:
        with open(os.path.join(data_dir, "all_proj_li.pkl"), "rb") as f:
            all_proj_li = load(f)
    return all_proj_li

def calc_potential_t(pos_t, speed_t):
    unique_pos = np.unique(pos_t)
    potential_t = []
    for pos in unique_pos:
        idx = pos_t <= pos
        potential_tx = np.sum(speed_t[idx])
        potential_t.append(potential_tx)
    return np.array(potential_t), unique_pos

def calculate_potential(proj,label):
    """
    Output:
        potential: list, len=#conditions, each element is a 2d array, shape=(#unique_pos, #time)
        unique_pos: nested list, len=#conditions, #time, each element is a 1d array, shape=(#unique_pos,)
        mean_act: 2d array, shape=(#conditions, # time)
    """
    cond = np.unique(label)
    potential = []
    unique_pos = []
    mean_act = []
    for c in cond:
        cond_mean = np.mean(proj[label==c], axis=0)
        cond_arr = proj[label==c]
        dxdt = np.diff(cond_arr, axis=1) * cond_mean[1:]
        cond_potential = []
        cond_unique_pos = []
        for t in range(dxdt.shape[1]):
            potential_t, unique_pos_t = calc_potential_t(cond_arr[:, t+1], dxdt[:, t])
            cond_potential.append(potential_t)
            cond_unique_pos.append(unique_pos_t)
        cond_potential = np.stack(cond_potential, axis=1)

        potential.append(cond_potential) 
        unique_pos.append(cond_unique_pos)
        mean_act.append(cond_mean)
    mean_act = np.stack(mean_act, axis=0)
    return potential, unique_pos, mean_act
    
def calc_all_potential(all_proj_li, rerun_calc=False):
    """
    Output:
        all_potential_li: list, len=#models, each element is a list, len=#reps, each element is a list, len=#conditions, each element is a 2d array, shape=(#unique_pos, #time)
        all_unique_pos_li: list, len=#models, each element is a list, len=#reps, each element is a nested list, len=#conditions, #time, each element is a 1d array, shape=(#unique_pos,)
        all_mean_act_li: list, len=#models, each element is a list, len=#reps, each element is 2d array, shape=(#conditions, # time)
    """
    if rerun_calc:
        all_potential_li = []
        all_unique_pos_li = []
        all_mean_act_li = []
        for m_idx, model_proj_li in enumerate(all_proj_li):
            model_potential_li = []
            model_unique_pos_li = []
            model_mean_act_li = []
            for rep in tqdm(range(len(model_proj_li))):
                proj = model_proj_li[rep]
                _, _, label = load_sac_act(f_dirs[m_idx], rep, reload=False)
                potential, unique_pos, mean_act = calculate_potential(proj, label)
                model_potential_li.append(potential)
                model_unique_pos_li.append(unique_pos)
                model_mean_act_li.append(mean_act)
            all_potential_li.append(model_potential_li)
            all_unique_pos_li.append(model_unique_pos_li)
            all_mean_act_li.append(model_mean_act_li)
        # save data
        with open(os.path.join(data_dir, "all_potential_li.pkl"), "wb") as f:
            dump(all_potential_li, f)
        with open(os.path.join(data_dir, "all_unique_pos_li.pkl"), "wb") as f:
            dump(all_unique_pos_li, f)
        with open(os.path.join(data_dir, "all_mean_act_li.pkl"), "wb") as f:
            dump(all_mean_act_li, f)
    else:
        with open(os.path.join(data_dir, "all_potential_li.pkl"), "rb") as f:
            all_potential_li = load(f)
        with open(os.path.join(data_dir, "all_unique_pos_li.pkl"), "rb") as f:
            all_unique_pos_li = load(f)
        with open(os.path.join(data_dir, "all_mean_act_li.pkl"), "rb") as f:
            all_mean_act_li = load(f)

if __name__ == "__main__":
    main()
