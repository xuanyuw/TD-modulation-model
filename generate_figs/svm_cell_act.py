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
from sklearn.model_selection import KFold 
from scipy.stats import sem

# plot settings
plt.rcParams["figure.figsize"] = [10, 4]
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2


f_dirs = ["crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
          "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model",
          "cutNonspec_model",
          "cutSpec_model"]
pic_dir = "./new_dyn_analysis/plots"
data_dir = "./new_dyn_analysis/data"

if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)
# model_type = f_dir
# model_type = f_dir.split("_")[-2]
# total_rep = 50
# total_shuf = 0
# all_lr = [2e-2]
# plot_sel = True
# rerun_calculation = True

# if "shuf" in f_dir:
#     plot_shuf = True
# else:
#     plot_shuf = False


def main():
    # plot_exp_var_ratio(f_dirs)
    run_pca_all_model(f_dirs, False)
    
    return

############## plot exp var ratio #####################
def get_all_exp_var_ratio(f_dir, total_rep=50):
    exp_var_ratio = []
    for rep in tqdm(range(total_rep)):
        mean_act_mat, _ = load_sac_act(f_dir, rep)
        mean_act_mat_ravel = ravel_3Dmat(mean_act_mat)
        pca = fit_PCA(mean_act_mat_ravel, n_components=10)
        exp_var_ratio.append(np.cumsum(pca.explained_variance_ratio_))
    return np.stack(exp_var_ratio, axis=0)

def plot_exp_var_ratio(f_dirs, pic_dir = pic_dir):
    plt.figure()
    for f_dir in f_dirs:
        model_type = f_dir.split("_")[-2]
        exp_var_ratio = get_all_exp_var_ratio(f_dir)
        plt.plot(np.mean(exp_var_ratio, axis=0), label=model_type)
        plt.errorbar(np.arange(10), np.mean(exp_var_ratio, axis=0), yerr=sem(exp_var_ratio, axis=0), fmt='none', capsize=3)
    plt.legend()
    plt.xlabel('# PC')
    plt.ylabel('Explained variance ratio')
    plt.title('Explained variance ratio for different models')
    plt.savefig(os.path.join(pic_dir, "exp_var_ratio.pdf"), format="pdf")
    plt.savefig(os.path.join(pic_dir, "exp_var_ratio.png"), format="png")
    plt.close()

############## run PCA on single trial #####################
def fit_PCA(act_mat, n_components=4):
    """
    Run PCA on the activity matrix
    """
    pca = PCA(n_components=n_components)
    pca.fit(act_mat)
    return pca

def run_pca_single_model(f_dir, rep, n_components=4):
    mean_act_mat, act_mat = load_sac_act(f_dir, rep)
    mean_act_mat_ravel = ravel_mat(mean_act_mat)
    pca = fit_PCA(mean_act_mat_ravel, n_components=4)

    act_mat_raveled = ravel_mat(act_mat)
    act_mat_pca = np.empty((act_mat_raveled.shape[0], n_components, act_mat_raveled.shape[-1]))
    for i in range(act_mat_raveled.shape[-1]):   
        act_mat_pca[:, :, i] = pca.transform(act_mat_raveled[:, :, i])
    return act_mat_pca

def run_pca_all_model(f_dirs, rerun_calc, total_rep=50, n_components=4):
    if rerun_calc:
        all_act_mat_pca = []
        for f_dir in f_dirs:
            act_mat_pca = []
            for rep in tqdm(range(total_rep)):
                act_mat_pca.append(run_pca_single_model(f_dir, rep, n_components=n_components))
            all_act_mat_pca.append(act_mat_pca)
        np.save(os.path.join(pic_dir, "all_act_mat_pca.npy"), all_act_mat_pca)
    else:
        all_act_mat_pca = np.load(os.path.join(pic_dir, "all_act_mat_pca.npy"), allow_pickle=True)
    return all_act_mat_pca


############## run SVM #####################


def run_SVM(X, y):
    clf = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)
    clf.fit(X, y)
    return clf

def run_k_folds(X, y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    acc = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.ravel(y[train_index]), np.ravel(y[test_index])
        clf = run_SVM(X_train, y_train)
        acc.append(clf.score(X_test, y_test))
    return np.mean(acc), np.std(acc)


############## tool functions #####################
def ravel_mat(m):
    if len(m.shape) == 3:
        new_m = []
        for i in range(m.shape[0]):
            new_m.append(np.ravel(m[i, :, :]))
        return np.stack(new_m, axis=0)
    elif len(m.shape) == 4:
        new_m = np.empty((m.shape[0], m.shape[1]*m.shape[2], m.shape[3]))
        for i in range(m.shape[0]):
            for j in range(m.shape[3]):
                new_m[i, :, j] = np.ravel(m[i, :, :, j])
        return new_m
    
def load_label(f_dir, rep, plot_correct = True):
    n = SimpleNamespace(
        **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
    )
    contra_idx_m1, ipsi_idx_m1 = find_sac_idx(n.y, True)
    contra_idx_m2, ipsi_idx_m2 = find_sac_idx(n.y, False)

    if plot_correct:
        contra_idx_m1 = combine_idx(contra_idx_m1, n.correct_idx)
        ipsi_idx_m1 = combine_idx(ipsi_idx_m1, n.correct_idx)
        contra_idx_m2 = combine_idx(contra_idx_m2, n.correct_idx)
        ipsi_idx_m2 = combine_idx(ipsi_idx_m2, n.correct_idx)

    

def load_sac_act(f_dir, rep, reload, module='targ', lr=0.02, normalize=True, plot_sel=True, plot_correct = True):
    """
    return mean activity: a 3D matrix (cell x time x sac_dir)
    and single trial activity: a 4D matrix (cell x time x sac_dir x rep)
    in sac_dir dimension: 0 = contra, 1 = ipsi
    """
    if reload:
        n = SimpleNamespace(
                **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
            )
        normalized_h = min_max_normalize(n.h)
        if normalize:
            h = normalized_h[20:, :, :]
        else:
            h = n.h[20:, :, :]

        h = np.swapaxes(h, 2, 0) # reshape activity to shape: cell x rep x time

        if plot_sel:
            saccade_selective = pick_selective_neurons(normalized_h, n.choice)
        else:
            saccade_selective = None    

        if plot_correct:
            c1_idx = combine_idx(n.correct_idx, n.choice)
            c2_idx = combine_idx(n.correct_idx, 1-n.choice)

            
        m1_rng = np.zeros(200)
        m1_rng[np.concatenate([np.arange(0, 80), np.arange(160, 180)])] = 1

        targ_rng = np.zeros(200)
        targ_rng[np.concatenate(
            (
                np.arange(40, 80),
                np.arange(120, 160),
                np.arange(170, 180),
                np.arange(190, 200),
            )
        )] = 1

        if module=='targ':
            selective_idx = combine_idx(targ_rng, saccade_selective)
        else:
            selective_idx = combine_idx(1-targ_rng, saccade_selective)

        c1_h = h[selective_idx,:, :][:, c1_idx, :]
        c2_h = h[selective_idx,:, :][:, c2_idx, :]
        mean_act_mat = np.stack((np.mean(c1_h, axis=1), np.mean(c2_h, axis=1)), -1)
    
        if c1_h.shape[1] <= c2_h.shape[1]:
            n_trial = c1_h.shape[1]
        else:
            n_trial = c2_h.shape[1]
        act_mat = np.stack((c1_h[:, :n_trial, :], c2_h[:, :n_trial, :]), -1).transpose(0, 2, 3, 1) # change shape to cell x time x sac_dir x rep
        dump((mean_act_mat, act_mat), open(os.path.join(data_dir, "sac_act_%s_rep%d.pkl" % (f_dir.split("_")[-2],rep)), "wb"))
    else:
        mean_act_mat, act_mat = load(os.path.join(data_dir, "sac_act_%s_rep%d.pkl" % (f_dir.split("_")[-2],rep)), "rb")
    # TODO: save labels and check if the above code is correct.
    return mean_act_mat, act_mat
    





if __name__ == "__main__":
    main()
