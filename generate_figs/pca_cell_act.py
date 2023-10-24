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

# plot settings
plt.rcParams["figure.figsize"] = [10, 4]
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2


f_dirs = ["crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
          "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback",
          "cutNonspec_model",
          "cutSpec_model"]
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

    load_sac_act(f_dirs[0], 0, calc_mean=True)






def load_sac_act(f_dir, rep, module='targ', lr=0.02, calc_mean=False, normalize=True, plot_sel=True):
    """
    if calc_mean is True, then return a 3D matrix: cell x time x sac_dir
    otherwise, return a 2D matrix: cell x (time x rep) x sac_dir
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

    if calc_mean:
        contra_mean = np.vstack([np.mean(contra_m1_h, axis=1), np.mean(contra_m2_h, axis=1)])
        ipsi_mean = np.vstack([np.mean(ipsi_m1_h, axis=1), np.mean(ipsi_m2_h, axis=1)])
        act_mat = np.stack((contra_mean, ipsi_mean), -1)
    else:
        #TODO: check how the original paper deal with unequal number of trials
        def ravel_3Dmat(m):
            new_m = []
            for i in range(m.shape[0]):
                new_m.append(np.ravel(m[i, :, :]))
            return np.stack(new_m, axis=0)
        
            

         
        


    


    

    

    return act_mat

    
def contruct_act_mat(h, y,):
    return

def fit_PCA(act_mat, n_components=2):
    """
    Run PCA on the activity matrix
    """
    pca = PCA(n_components=n_components)
    pca.fit(act_mat)
    return pca




if __name__ == "__main__":
    main()
