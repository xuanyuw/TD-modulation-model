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

if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def main():
    # plot_exp_var_ratio(f_dirs)
    all_act_mat_pca = run_pca_all_model(f_dirs, True)
    all_model_acc_li, _, _ = run_SVM_all_model(
        f_dirs, all_act_mat_pca, rerun_calc=True
    )
    plot_svm_acc(all_model_acc_li)
    return


############## plot exp var ratio #####################
def get_all_exp_var_ratio(f_dir, total_rep=50, n_components=20, reload_data=False):
    exp_var_ratio = []
    for rep in tqdm(range(total_rep)):
        mean_act_mat, _, _ = load_sac_act(f_dir, rep, reload=reload_data)
        # mean_act_mat_ravel = ravel_mat(mean_act_mat)
        pca = fit_PCA(mean_act_mat, n_components=n_components)
        exp_var_ratio.append(np.cumsum(pca.explained_variance_ratio_))
    return np.stack(exp_var_ratio, axis=0)


def plot_exp_var_ratio(f_dirs, pic_dir=pic_dir, n_components=20):
    plt.figure()
    for f_dir in f_dirs:
        model_type = f_dir.split("_")[-2]
        exp_var_ratio = get_all_exp_var_ratio(f_dir)
        plt.plot(np.mean(exp_var_ratio, axis=0), label=model_type)
        plt.errorbar(
            np.arange(n_components),
            np.mean(exp_var_ratio, axis=0),
            yerr=sem(exp_var_ratio, axis=0),
            fmt="none",
            capsize=3,
        )
    plt.legend()
    plt.xlabel("# PC")
    plt.ylabel("Explained variance ratio")
    plt.title("Explained variance ratio for different models")
    plt.savefig(os.path.join(pic_dir, "exp_var_ratio.pdf"), format="pdf")
    plt.savefig(os.path.join(pic_dir, "exp_var_ratio.png"), format="png")
    plt.close()


############## run PCA on single trial #####################
def fit_PCA(act_mat, n_components=3):
    """
    Run PCA on the activity matrix
    """
    pca = PCA(n_components=n_components)
    pca.fit(act_mat)
    return pca


def run_pca_single_model(f_dir, rep, n_components=3, reload_data=False):
    mean_act_mat, act_mat, _ = load_sac_act(f_dir, rep, reload=reload_data)
    pca = fit_PCA(mean_act_mat, n_components=3)
    act_mat_pca = np.empty((n_components, act_mat.shape[1], act_mat.shape[-1]))
    for trial in range(act_mat.shape[1]):
        for t in range(act_mat.shape[-1]):
            act_mat_pca[:, trial, t] = pca.transform(
                act_mat[:, trial, t].reshape(1, -1)
            )

    # act_mat_raveled = ravel_mat(act_mat)
    # act_mat_pca = np.empty((act_mat_raveled.shape[0], n_components, act_mat_raveled.shape[-1]))
    # for i in range(act_mat_raveled.shape[-1]):
    #     act_mat_pca[:, :, i] = pca.transform(act_mat_raveled[:, :, i])
    return act_mat_pca


def run_pca_all_model(f_dirs, rerun_calc, total_rep=50, n_components=3):
    all_act_mat_pca = []
    for f_dir in f_dirs:
        if rerun_calc:
            act_mat_pca = []
            for rep in tqdm(range(total_rep)):
                act_mat_pca.append(
                    run_pca_single_model(f_dir, rep, n_components=n_components)
                )
            dump(
                act_mat_pca,
                open(
                    os.path.join(
                        data_dir, "act_mat_pca_%s.npy" % (f_dir.split("_")[-2])
                    ),
                    "wb",
                ),
            )
        else:
            act_mat_pca = load(
                open(
                    os.path.join(
                        data_dir, "act_mat_pca_%s.npy" % (f_dir.split("_")[-2])
                    ),
                    "rb",
                )
            )
        all_act_mat_pca.append(act_mat_pca)
    return all_act_mat_pca


############## run SVM #####################


def run_SVM_all_model(
    f_dirs, act_all_pca, rerun_calc, total_rep=50, n_components=3, k=10
):
    all_model_acc_li = []
    all_coef_li = []
    all_intercept_li = []
    for i, f_dir in enumerate(f_dirs):
        if rerun_calc:
            model_acc = []
            model_coef_li = []
            model_intercept_li = []
            for rep in tqdm(range(total_rep)):
                act_mat = act_all_pca[i][rep]
                _, _, label = load_sac_act(f_dir, rep, True)
                acc_t = []
                coef_t = []
                intercept_t = []
                # ste_t = np.zeros((act_mat.shape[-1]))
                for t in range(act_mat.shape[-1]):
                    # run SVM with cross validation
                    clf = LinearSVC(random_state=0)
                    cv_result = cross_validate(
                        clf, act_mat[:, :, t].T, label, cv=k, return_estimator=True
                    )
                    clf_li = cv_result["estimator"]
                    acc_t.append(
                        cv_result["test_score"]
                    )  # save the accuracy of the fitted SVM model at each time point t
                    # save the coefficient and the intercept of the fitted SVM model
                    coef = np.vstack([x.coef_ for x in clf_li])
                    intercept = np.vstack([x.intercept_ for x in clf_li])
                    coef_t.append(coef)
                    intercept_t.append(intercept)
                acc_t = np.stack(
                    acc_t, axis=0
                )  # the shape of acc_t is: # time points x # CV
                coef_t = np.stack(
                    coef_t, axis=0
                )  # the shape of coef_t is: # time points x # CV x n_components
                intercept_t = np.squeeze(
                    np.stack(intercept_t, axis=0)
                )  # the shape of intercept_t is # time points x # CV
                model_acc.append(acc_t)  # save the acurracy for each model repetition
                model_intercept_li.append(intercept_t)
                model_coef_li.append(coef_t)
            model_acc = np.stack(
                model_acc, axis=-1
            )  # the shape of all_acc is: # time points x # CV x # model reps
            model_coef_li = np.stack(
                model_coef_li, axis=-1
            )  # the shape of all_coef is: # time points x # CV x n_components x # model reps
            model_intercept_li = np.stack(
                model_intercept_li, axis=-1
            )  # the shape of all_intercept is: # time points x # CV x # model reps
            dump(
                (model_acc, model_coef_li, model_intercept_li),
                open(
                    os.path.join(
                        data_dir, "all_svm_acc_%s.npy" % (f_dir.split("_")[-2])
                    ),
                    "wb",
                ),
            )
        else:
            model_acc, model_coef_li, model_intercept_li = load(
                open(
                    os.path.join(
                        data_dir, "all_svm_acc_%s.npy" % (f_dir.split("_")[-2])
                    ),
                    "rb",
                )
            )
        all_model_acc_li.append(model_acc)
        all_coef_li.append(model_coef_li)
        all_intercept_li.append(model_intercept_li)
    return all_model_acc_li, all_coef_li, all_intercept_li


def plot_svm_acc(all_model_acc_li):
    m_types = [f_dir.split("_")[-2] for f_dir in f_dirs]
    plt.figure()
    for i, m_type in enumerate(m_types):
        plt.plot(np.mean(all_model_acc_li[i], axis=(1, 2)), label=m_type)
        plt.errorbar(
            np.arange(all_model_acc_li[i].shape[0]),
            np.mean(all_model_acc_li[i], axis=(1, 2)),
            yerr=sem(np.mean(all_model_acc_li[i], axis=1), axis=-1),
            fmt="none",
            capsize=3,
            color=plt.gca().lines[-1].get_color(),
        )
    ylim = plt.gca().get_ylim()
    plt.vlines(20, min(ylim), max(ylim), linestyles="dashed", colors="k")
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Accuracy")
    plt.title("SVM accuracy for different models")
    plt.savefig(os.path.join(pic_dir, "svm_acc.pdf"), format="pdf")
    plt.savefig(os.path.join(pic_dir, "svm_acc.png"), format="png")


############## tool functions #####################
def ravel_mat(m):
    if len(m.shape) == 3:
        new_m = []
        for i in range(m.shape[0]):
            new_m.append(np.ravel(m[i, :, :]))
        return np.stack(new_m, axis=0)
    elif len(m.shape) == 4:
        new_m = np.empty((m.shape[0], m.shape[1] * m.shape[2], m.shape[3]))
        for i in range(m.shape[0]):
            for j in range(m.shape[3]):
                new_m[i, :, j] = np.ravel(m[i, :, :, j])
        return new_m


def load_sac_act(
    f_dir,
    rep,
    reload,
    module="targ",
    lr=0.02,
    normalize=True,
    plot_sel=True,
    plot_correct=True,
):
    """
    return:
    mean activity matrix with shape (# coh x # stim x # choice x time, cell)
    single trial activity: a 3D matrix (cell x rep x time)
    the label choice
    """
    if reload:
        n = SimpleNamespace(
            **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
        )
        normalized_h = min_max_normalize(n.h)
        if normalize:
            temp_h = normalized_h[20:, :, :]
        else:
            temp_h = n.h[20:, :, :]

        temp_h = np.swapaxes(
            temp_h, 2, 0
        )  # reshape activity to shape: cell x rep x time

        if plot_sel:
            saccade_selective = pick_selective_neurons(normalized_h, n.choice)
        else:
            saccade_selective = None

        if plot_correct:
            h = temp_h[:, n.correct_idx, :]
            label = n.choice[n.correct_idx]
        else:
            h = temp_h
            label = n.choice

        targ_rng = np.zeros(200)
        targ_rng[
            np.concatenate(
                (
                    np.arange(40, 80),
                    np.arange(120, 160),
                    np.arange(170, 180),
                    np.arange(190, 200),
                )
            )
        ] = 1

        if module == "targ":
            selective_idx = combine_idx(targ_rng, saccade_selective)
        else:
            selective_idx = combine_idx(1 - targ_rng, saccade_selective)
        act_mat = h[selective_idx, :, :]

        coh_dict = find_coh_idx(n.stim_level)
        mean_act_mat = []
        for coh in coh_dict.keys():
            for stim in np.unique(n.stim_dir):
                for choice in np.unique(n.choice):
                    temp_idx = combine_idx(
                        coh_dict[coh], n.stim_dir == stim, n.choice == choice
                    )
                    if plot_correct:
                        temp_idx = combine_idx(temp_idx, n.correct_idx)
                    if sum(temp_idx) > 0:
                        mean_act_mat.append(
                            np.mean(temp_h[selective_idx, :, :][:, temp_idx, :], axis=1)
                        )

        mean_act_mat = np.swapaxes(np.stack(mean_act_mat, axis=0), 0, 1)
        mean_act_mat = ravel_mat(mean_act_mat).T

        dump(
            (mean_act_mat, act_mat, label),
            open(
                os.path.join(
                    data_dir, "sac_act_%s_rep%d.pkl" % (f_dir.split("_")[-2], rep)
                ),
                "wb",
            ),
        )
    else:
        with open(
            os.path.join(
                data_dir, "sac_act_%s_rep%d.pkl" % (f_dir.split("_")[-2], rep)
            ),
            "rb",
        ) as f:
            mean_act_mat, act_mat, label = load(f)
    return mean_act_mat, act_mat, label


if __name__ == "__main__":
    main()
