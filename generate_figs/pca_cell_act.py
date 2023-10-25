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
pic_dir = "./pca_plots"
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
    run_pca_single_model(f_dirs[0], 0)
    
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
def run_pca_single_model(f_dir, rep):
    mean_act_mat, act_mat = load_sac_act(f_dir, rep)
    mean_act_mat_ravel = ravel_3Dmat(mean_act_mat)
    pca = fit_PCA(mean_act_mat_ravel)
    # TODO: figure out how to transform the single trial activity by time points
    pca.transform
    # for i in range():
    
    return pca.explained_variance_ratio_




def ravel_3Dmat(m):
    new_m = []
    for i in range(m.shape[0]):
        new_m.append(np.ravel(m[i, :, :]))
    return np.stack(new_m, axis=0)


def load_sac_act(f_dir, rep, module='targ', lr=0.02, normalize=True, plot_sel=True, plot_correct = True):
    """
    return mean activity: a 3D matrix (cell x time x sac_dir)
    and single trial activity: a 4D matrix (cell x rep  x time x sac_dir)
    in sac_dir dimension: 0 = contra, 1 = ipsi
    """

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

    contra_idx_m1, ipsi_idx_m1 = find_sac_idx(n.y, True)
    contra_idx_m2, ipsi_idx_m2 = find_sac_idx(n.y, False)

    if plot_correct:
        contra_idx_m1 = combine_idx(contra_idx_m1, n.correct_idx)
        ipsi_idx_m1 = combine_idx(ipsi_idx_m1, n.correct_idx)
        contra_idx_m2 = combine_idx(contra_idx_m2, n.correct_idx)
        ipsi_idx_m2 = combine_idx(ipsi_idx_m2, n.correct_idx)

        

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
        m1_selective = combine_idx(targ_rng, m1_rng, saccade_selective)
        m2_selective = combine_idx(targ_rng, 1 - m1_rng, saccade_selective)
    else:
        m1_selective = combine_idx(1-targ_rng, m1_rng, saccade_selective)
        m2_selective = combine_idx(1-targ_rng, 1 - m1_rng, saccade_selective)

    contra_m1_h = h[m1_selective, :, :][:, contra_idx_m1, :]
    ipsi_m1_h = h[m1_selective,:, :][:, ipsi_idx_m1, :]
    contra_m2_h = h[m2_selective,:, :][:, contra_idx_m2, :]
    ipsi_m2_h = h[m2_selective,:, :][:, ipsi_idx_m2, :]

    contra_mean = np.vstack([np.mean(contra_m1_h, axis=1), np.mean(contra_m2_h, axis=1)])
    ipsi_mean = np.vstack([np.mean(ipsi_m1_h, axis=1), np.mean(ipsi_m2_h, axis=1)])
    mean_act_mat = np.stack((contra_mean, ipsi_mean), -1)
   
    if contra_m1_h.shape[1] <= ipsi_m1_h.shape[1]:
        n_trial = contra_m1_h.shape[1]
    else:
        n_trial = ipsi_m1_h.shape[1]

    contra_act = np.vstack([contra_m1_h[:, :n_trial, :], contra_m2_h[:, :n_trial, :]])
    ipsi_act = np.vstack([ipsi_m1_h[:, :n_trial, :], ipsi_m2_h[:, :n_trial, :]])
    act_mat = np.stack((contra_act, ipsi_act), -1)

    return mean_act_mat, act_mat
    


def fit_PCA(act_mat, n_components=4):
    """
    Run PCA on the activity matrix
    """
    pca = PCA(n_components=n_components)
    pca.fit(act_mat)
    return pca




if __name__ == "__main__":
    main()
