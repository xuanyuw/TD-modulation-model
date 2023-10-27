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

from svm_cell_act import run_pca_all_model, run_SVM_all_model

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
        f_dirs, 
        all_act_mat_pca, rerun_calc=False
    )
    all_proj_li = project_all_data(all_act_mat_pca, all_coef_li, all_intercept_li, all_model_acc_li, total_rep=50)

def find_choice_axis(coefficients, intercept):
    """
    get the vector perpendicular to the SVM hyperplane as the choice axis
    """
    # TODO: correct orthogonal vector calculation
    vector = np.array(coefficients + [intercept])
    random_vector = np.random.randn(*vector.shape)
    orthogonal_vector = random_vector - np.dot(random_vector, vector) * vector
    return orthogonal_vector

def project_to_choice_axis(perpendicular_vector, data):
    """
    project data to the choice axis
    """
    return np.dot(data, perpendicular_vector)

def project_all_data(all_act_mat_pca, all_coef_li, all_intercept_li, all_model_acc_li, total_rep=50):
    """
    project all data to the choice axis
    """
    all_proj_li = []
    for coef, intercept, act_mat_pca, acc in zip(all_coef_li, all_intercept_li, all_act_mat_pca, all_model_acc_li):
        for rep in range(total_rep):
            model_act_mat = act_mat_pca[rep]
            # use best performing of the 10 CV results as the choice axis
            best_model_idx = np.argmax(acc[-1, :, rep])
            model_intercept = intercept[-1, best_model_idx, rep]
            model_coef = coef[-1, best_model_idx, :, rep]
            perpendicular_vector = find_choice_axis(model_coef, model_intercept)
            proj = project_to_choice_axis(perpendicular_vector, act_mat_pca)
            all_proj_li.append(proj)
    return all_proj_li

def calculate_potential():
    return


if __name__ == "__main__":
    main()