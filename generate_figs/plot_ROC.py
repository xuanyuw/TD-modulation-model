import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import (
    find_coh_idx,
    find_sac_idx,
    combine_idx,
    load_test_data,
    min_max_normalize,
    get_pref_idx,
    pick_selective_neurons,
)
from types import SimpleNamespace
from pickle import dump, load
from tqdm import tqdm
from time import perf_counter
from scipy.stats import ttest_rel, sem, ttest_1samp
from joblib import Parallel, delayed
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# plot settings

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({"font.size": 15})
mpl.rcParams["lines.linewidth"] = 2


f_dir = "F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
total_rep = 50
# f_dir = "/Users/xuanyuwu/Documents/GitHub/TD-modulation-model/crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
all_rep = range(total_rep)
lr = 0.02
n_jobs = 8

stim_st_time = 45
target_st_time = 25
rerun_calc = False
normalize = False
sep_sac = True
plot_sel = True

# if not sep_sac:
plt.rcParams["figure.figsize"] = [6, 4]
# else:
# plt.rcParams['figure.figsize'] = [10, 4]

h_len = 70 - 19 - 3


fn = os.path.join(f_dir, "sep_sac_ROC_dir_%dnet.pkl" % total_rep)
fn_dir = os.path.join(f_dir, "all_ROC_dir_%dnet.pkl" % total_rep)
fn_sac = os.path.join(f_dir, "all_ROC_sac_%dnet.pkl" % total_rep)


def main():
    motion_rng = np.concatenate(
        (np.arange(0, 40), np.arange(80, 120), np.arange(160, 170), np.arange(180, 190))
    )
    target_rng = np.concatenate(
        (
            np.arange(40, 80),
            np.arange(120, 160),
            np.arange(170, 180),
            np.arange(190, 200),
        )
    )
    m1_rng = np.concatenate(
        (np.arange(0, 40), np.arange(80, 90))
    )  # range of m1 neuron indices after separated by RF
    if rerun_calc:
        if sep_sac:
            H_ipsi_dir_ROC = np.empty((h_len, 100 * len(all_rep))) * np.nan
            H_contra_dir_ROC = np.empty((h_len, 100 * len(all_rep))) * np.nan
            M_ipsi_dir_ROC = np.empty((h_len, 100 * len(all_rep))) * np.nan
            M_contra_dir_ROC = np.empty((h_len, 100 * len(all_rep))) * np.nan
            L_ipsi_dir_ROC = np.empty((h_len, 100 * len(all_rep))) * np.nan
            L_contra_dir_ROC = np.empty((h_len, 100 * len(all_rep))) * np.nan
            Z_ipsi_dir_ROC = np.empty((h_len, 100 * len(all_rep))) * np.nan
            Z_contra_dir_ROC = np.empty((h_len, 100 * len(all_rep))) * np.nan

        else:
            H_dir_ROC = np.empty((len(all_rep), h_len, 100)) * np.nan
            M_dir_ROC = np.empty((len(all_rep), h_len, 100)) * np.nan
            L_dir_ROC = np.empty((len(all_rep), h_len, 100)) * np.nan
            Z_dir_ROC = np.empty((len(all_rep), h_len, 100)) * np.nan
            H_sac_ROC = np.empty((len(all_rep), h_len, 100)) * np.nan
            M_sac_ROC = np.empty((len(all_rep), h_len, 100)) * np.nan
            L_sac_ROC = np.empty((len(all_rep), h_len, 100)) * np.nan
            Z_sac_ROC = np.empty((len(all_rep), h_len, 100)) * np.nan
        if not sep_sac:
            pbar = tqdm(total=len(all_rep) * 2 * 4)
        else:
            pbar = tqdm(total=len(all_rep) * 4)
        idx = 0
        for rep in all_rep:
            # print('Running ROC calculation for rep %d ... '%rep)
            n = SimpleNamespace(
                **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
            )
            normalized_h = min_max_normalize(n.h)
            if normalize:
                h = normalized_h
            else:
                h = n.h

            coh_dict = find_coh_idx(n.stim_level)
            H_idx = coh_dict["H"]
            M_idx = coh_dict["M"]
            L_idx = coh_dict["L"]
            Z_idx = coh_dict["Z"]
            pref_dir, pref_sac = get_pref_idx(n, h)
            if sep_sac:
                (
                    H_ipsi_dir_ROC[:, rep * 100 : (rep + 1) * 100],
                    H_contra_dir_ROC[:, rep * 100 : (rep + 1) * 100],
                ) = calc_sac_sep_ROC(
                    h[17:, :, motion_rng], n, m1_rng, H_idx, pref_dir[:, motion_rng]
                )
                pbar.update(1)
                (
                    M_ipsi_dir_ROC[:, rep * 100 : (rep + 1) * 100],
                    M_contra_dir_ROC[:, rep * 100 : (rep + 1) * 100],
                ) = calc_sac_sep_ROC(
                    h[17:, :, motion_rng], n, m1_rng, M_idx, pref_dir[:, motion_rng]
                )
                pbar.update(1)
                (
                    L_ipsi_dir_ROC[:, rep * 100 : (rep + 1) * 100],
                    L_contra_dir_ROC[:, rep * 100 : (rep + 1) * 100],
                ) = calc_sac_sep_ROC(
                    h[17:, :, motion_rng], n, m1_rng, L_idx, pref_dir[:, motion_rng]
                )
                pbar.update(1)
                (
                    Z_ipsi_dir_ROC[:, rep * 100 : (rep + 1) * 100],
                    Z_contra_dir_ROC[:, rep * 100 : (rep + 1) * 100],
                ) = calc_sac_sep_ROC(
                    h[17:, :, motion_rng], n, m1_rng, Z_idx, pref_dir[:, motion_rng]
                )
                pbar.update(1)
            else:
                H_dir_ROC[rep, :, :] = calc_ROC(
                    h[17:, :, motion_rng], n, H_idx, pref_dir[:, motion_rng]
                )
                pbar.update(1)
                M_dir_ROC[rep, :, :] = calc_ROC(
                    h[17:, :, motion_rng], n, M_idx, pref_dir[:, motion_rng]
                )
                pbar.update(1)
                L_dir_ROC[rep, :, :] = calc_ROC(
                    h[17:, :, motion_rng], n, L_idx, pref_dir[:, motion_rng]
                )
                pbar.update(1)
                Z_dir_ROC[rep, :, :] = calc_ROC(
                    h[17:, :, motion_rng], n, Z_idx, pref_dir[:, motion_rng]
                )
                pbar.update(1)

                H_sac_ROC[rep, :, :] = calc_ROC(
                    h[17:, :, target_rng], n, H_idx, pref_sac[:, target_rng]
                )
                pbar.update(1)
                M_sac_ROC[rep, :, :] = calc_ROC(
                    h[17:, :, target_rng], n, M_idx, pref_sac[:, target_rng]
                )
                pbar.update(1)
                L_sac_ROC[rep, :, :] = calc_ROC(
                    h[17:, :, target_rng], n, L_idx, pref_sac[:, target_rng]
                )
                pbar.update(1)
                Z_sac_ROC[rep, :, :] = calc_ROC(
                    h[17:, :, target_rng], n, Z_idx, pref_sac[:, target_rng]
                )
                pbar.update(1)

        pbar.close()
        if sep_sac:
            with open(fn, "wb") as f:
                dump(
                    [
                        H_ipsi_dir_ROC,
                        H_contra_dir_ROC,
                        M_ipsi_dir_ROC,
                        M_contra_dir_ROC,
                        L_ipsi_dir_ROC,
                        L_contra_dir_ROC,
                        Z_ipsi_dir_ROC,
                        Z_contra_dir_ROC,
                    ],
                    f,
                )
        else:
            with open(fn_dir, "wb") as f:
                dump([H_dir_ROC, M_dir_ROC, L_dir_ROC, Z_dir_ROC], f)
            with open(fn_sac, "wb") as f:
                dump([H_sac_ROC, M_sac_ROC, L_sac_ROC, Z_sac_ROC], f)

    else:
        with open(fn, "rb") as f:
            (
                H_ipsi_dir_ROC,
                H_contra_dir_ROC,
                M_ipsi_dir_ROC,
                M_contra_dir_ROC,
                L_ipsi_dir_ROC,
                L_contra_dir_ROC,
                Z_ipsi_dir_ROC,
                Z_contra_dir_ROC,
            ) = load(f)
        with open(fn_dir, "rb") as f:
            [H_dir_ROC, M_dir_ROC, L_dir_ROC, Z_dir_ROC] = load(f)
        with open(fn_sac, "rb") as f:
            [H_sac_ROC, M_sac_ROC, L_sac_ROC, Z_sac_ROC] = load(f)

    f = open(
        os.path.join("generate_figs", "Fig5", "stat_test_%dnet.txt" % total_rep), "w"
    )
    sys.stdout = f

    if not plot_sel:
        plot_all_avg_ROC(H_dir_ROC, M_dir_ROC, L_dir_ROC, Z_dir_ROC, "dir")
        plot_all_avg_ROC(H_sac_ROC, M_sac_ROC, L_sac_ROC, Z_sac_ROC, "sac")
    else:
        motion_selective, saccade_selective = get_sel_cells()

        H_dir_ROC_sel = np.empty((len(all_rep), h_len, 100)) * np.nan
        M_dir_ROC_sel = np.empty((len(all_rep), h_len, 100)) * np.nan
        L_dir_ROC_sel = np.empty((len(all_rep), h_len, 100)) * np.nan
        Z_dir_ROC_sel = np.empty((len(all_rep), h_len, 100)) * np.nan
        H_sac_ROC_sel = np.empty((len(all_rep), h_len, 100)) * np.nan
        M_sac_ROC_sel = np.empty((len(all_rep), h_len, 100)) * np.nan
        L_sac_ROC_sel = np.empty((len(all_rep), h_len, 100)) * np.nan
        Z_sac_ROC_sel = np.empty((len(all_rep), h_len, 100)) * np.nan

        for rep in all_rep:
            m_sel = motion_selective[rep, motion_rng].astype(bool)
            s_sel = saccade_selective[rep, target_rng].astype(bool)

            H_dir_ROC_sel[rep, :, : sum(m_sel)] = H_dir_ROC[rep][:, m_sel]
            M_dir_ROC_sel[rep, :, : sum(m_sel)] = M_dir_ROC[rep][:, m_sel]
            L_dir_ROC_sel[rep, :, : sum(m_sel)] = L_dir_ROC[rep][:, m_sel]
            Z_dir_ROC_sel[rep, :, : sum(m_sel)] = Z_dir_ROC[rep][:, m_sel]
            H_sac_ROC_sel[rep, :, : sum(s_sel)] = H_sac_ROC[rep][:, s_sel]
            M_sac_ROC_sel[rep, :, : sum(s_sel)] = M_sac_ROC[rep][:, s_sel]
            L_sac_ROC_sel[rep, :, : sum(s_sel)] = L_sac_ROC[rep][:, s_sel]
            Z_sac_ROC_sel[rep, :, : sum(s_sel)] = Z_sac_ROC[rep][:, s_sel]
        plot_all_avg_ROC(
            H_dir_ROC_sel, M_dir_ROC_sel, L_dir_ROC_sel, Z_dir_ROC_sel, "dir"
        )
        plot_all_avg_ROC(
            H_sac_ROC_sel, M_sac_ROC_sel, L_sac_ROC_sel, Z_sac_ROC_sel, "sac"
        )

    if not plot_sel:
        line_dict = {
            "H_ipsi": H_ipsi_dir_ROC,
            "H_contra": H_contra_dir_ROC,
            "M_ipsi": M_ipsi_dir_ROC,
            "M_contra": M_contra_dir_ROC,
            "L_ipsi": L_ipsi_dir_ROC,
            "L_contra": L_contra_dir_ROC,
            "Z_ipsi": Z_ipsi_dir_ROC,
            "Z_contra": Z_contra_dir_ROC,
        }
        plot_all_avg_ROC_sep_sac(line_dict)
    else:
        motion_selective, _ = get_sel_cells()
        H_ipsi_dir_ROC_sel = np.empty((h_len, 100 * len(all_rep))) * np.nan
        H_contra_dir_ROC_sel = np.empty((h_len, 100 * len(all_rep))) * np.nan
        M_ipsi_dir_ROC_sel = np.empty((h_len, 100 * len(all_rep))) * np.nan
        M_contra_dir_ROC_sel = np.empty((h_len, 100 * len(all_rep))) * np.nan
        L_ipsi_dir_ROC_sel = np.empty((h_len, 100 * len(all_rep))) * np.nan
        L_contra_dir_ROC_sel = np.empty((h_len, 100 * len(all_rep))) * np.nan
        Z_ipsi_dir_ROC_sel = np.empty((h_len, 100 * len(all_rep))) * np.nan
        Z_contra_dir_ROC_sel = np.empty((h_len, 100 * len(all_rep))) * np.nan

        idx = 0
        for rep in all_rep:
            m_sel = motion_selective[rep, motion_rng].astype(bool)

            H_ipsi_dir_ROC_temp = H_ipsi_dir_ROC[:, rep * 100 : (rep + 1) * 100]
            H_contra_dir_ROC_temp = H_contra_dir_ROC[:, rep * 100 : (rep + 1) * 100]
            M_ipsi_dir_ROC_temp = M_ipsi_dir_ROC[:, rep * 100 : (rep + 1) * 100]
            M_contra_dir_ROC_temp = M_contra_dir_ROC[:, rep * 100 : (rep + 1) * 100]
            L_ipsi_dir_ROC_temp = L_ipsi_dir_ROC[:, rep * 100 : (rep + 1) * 100]
            L_contra_dir_ROC_temp = L_contra_dir_ROC[:, rep * 100 : (rep + 1) * 100]
            Z_ipsi_dir_ROC_temp = Z_ipsi_dir_ROC[:, rep * 100 : (rep + 1) * 100]
            Z_contra_dir_ROC_temp = Z_contra_dir_ROC[:, rep * 100 : (rep + 1) * 100]

            H_ipsi_dir_ROC_sel[:, idx : idx + sum(m_sel)] = H_ipsi_dir_ROC_temp[
                :, m_sel
            ]
            H_contra_dir_ROC_sel[:, idx : idx + sum(m_sel)] = H_contra_dir_ROC_temp[
                :, m_sel
            ]
            M_ipsi_dir_ROC_sel[:, idx : idx + sum(m_sel)] = M_ipsi_dir_ROC_temp[
                :, m_sel
            ]
            M_contra_dir_ROC_sel[:, idx : idx + sum(m_sel)] = M_contra_dir_ROC_temp[
                :, m_sel
            ]
            L_ipsi_dir_ROC_sel[:, idx : idx + sum(m_sel)] = L_ipsi_dir_ROC_temp[
                :, m_sel
            ]
            L_contra_dir_ROC_sel[:, idx : idx + sum(m_sel)] = L_contra_dir_ROC_temp[
                :, m_sel
            ]
            Z_ipsi_dir_ROC_sel[:, idx : idx + sum(m_sel)] = Z_ipsi_dir_ROC_temp[
                :, m_sel
            ]
            Z_contra_dir_ROC_sel[:, idx : idx + sum(m_sel)] = Z_contra_dir_ROC_temp[
                :, m_sel
            ]
        line_dict = {
            "H_ipsi": H_ipsi_dir_ROC_sel,
            "H_contra": H_contra_dir_ROC_sel,
            "M_ipsi": M_ipsi_dir_ROC_sel,
            "M_contra": M_contra_dir_ROC_sel,
            "L_ipsi": L_ipsi_dir_ROC_sel,
            "L_contra": L_contra_dir_ROC_sel,
            "Z_ipsi": Z_ipsi_dir_ROC_sel,
            "Z_contra": Z_contra_dir_ROC_sel,
        }
        plot_all_avg_ROC_sep_sac(line_dict)
    f.close()


def get_sel_cells():
    if not os.path.exists(
        os.path.join(f_dir, "motion_selective_cell_idx.npy")
    ) or not os.path.exists(os.path.join(f_dir, "saccade_selective_cell_idx.npy")):
        motion_selective = np.zeros((len(all_rep), 200))
        saccade_selective = np.zeros((len(all_rep), 200))
        for rep in all_rep:
            # print('Running ROC calculation for rep %d ... '%rep)
            n = SimpleNamespace(
                **load_test_data(f_dir, "test_output_lr%f_rep%d.h5" % (lr, rep))
            )
            normalized_h = min_max_normalize(n.h)
            if plot_sel:
                motion_selective[rep, :] = pick_selective_neurons(
                    normalized_h, n.stim_dir
                )
                saccade_selective[rep, :] = pick_selective_neurons(
                    normalized_h, n.choice
                )
        with open(os.path.join(f_dir, "motion_selective_cell_idx.npy"), "wb") as f:
            np.save(f, motion_selective)
        with open(os.path.join(f_dir, "saccade_selective_cell_idx.npy"), "wb") as f:
            np.save(f, saccade_selective)
    else:
        with open(os.path.join(f_dir, "motion_selective_cell_idx.npy"), "rb") as f:
            motion_selective = np.load(f)
        with open(os.path.join(f_dir, "saccade_selective_cell_idx.npy"), "rb") as f:
            saccade_selective = np.load(f)
    return motion_selective, saccade_selective


def rocN(x, y, N=100):
    x = x.flatten("F")
    y = y.flatten("F")
    zlo = min(min(x), min(y))
    zhi = max(max(x), max(y))
    z = np.linspace(zlo, zhi, N)
    fa = np.zeros((N,))
    hit = np.zeros((N,))
    for i in range(N):
        fa[N - (i + 1)] = sum(y > z[i])
        hit[N - (i + 1)] = sum(x > z[i])

    fa = fa / y.shape[0]
    hit = hit / x.shape[0]
    a = np.trapz(y=hit, x=fa)

    return a


def calc_ROC(h, n, coh_idx, pref_idx):
    # all_ROC = np.zeros((h.shape[0]-5, h.shape[2]))
    def cell_roc(i):
        roc = np.zeros((h.shape[0] - 5,))
        pre_idx = combine_idx(pref_idx[:, i], n.correct_idx, coh_idx)
        non_idx = combine_idx(~pref_idx[:, i], n.correct_idx, coh_idx)
        for j in range(h.shape[0] - 5):
            h_pre = np.mean(h[j : j + 5, pre_idx, i], axis=0)
            h_non = np.mean(h[j : j + 5, non_idx, i], axis=0)
            if len(h_pre) == 0 or len(h_non) == 0:
                roc[j] = np.nan
            else:
                roc[j] = rocN(h_pre, h_non)

        return roc

    all_ROC = Parallel(n_jobs=n_jobs)(delayed(cell_roc)(i) for i in range(h.shape[2]))
    return np.array(all_ROC).T


def calc_sac_sep_ROC(h, n, m1_rng, coh_idx, pref_idx):
    contra_idx_m1, ipsi_idx_m1 = find_sac_idx(n.y, True)
    contra_idx_m2, ipsi_idx_m2 = find_sac_idx(n.y, False)

    ipsi_ROC = np.zeros((h.shape[0] - 5, h.shape[2]))
    contra_ROC = np.zeros((h.shape[0] - 5, h.shape[2]))
    for i in range(h.shape[2]):
        if i in m1_rng:
            contra_idx = contra_idx_m1
            ipsi_idx = ipsi_idx_m1
        else:
            contra_idx = contra_idx_m2
            ipsi_idx = ipsi_idx_m2
        ipsi_pref_idx = combine_idx(ipsi_idx, pref_idx[:, i], n.correct_idx, coh_idx)
        contra_pref_idx = combine_idx(
            contra_idx, pref_idx[:, i], n.correct_idx, coh_idx
        )
        ipsi_non_idx = combine_idx(ipsi_idx, ~pref_idx[:, i], n.correct_idx, coh_idx)
        contra_non_idx = combine_idx(
            contra_idx, ~pref_idx[:, i], n.correct_idx, coh_idx
        )
        for j in range(h.shape[0] - 5):
            h_ipsi_pref = np.mean(h[j : j + 5, ipsi_pref_idx, i], axis=0)
            h_contra_pref = np.mean(h[j : j + 5, contra_pref_idx, i], axis=0)
            h_ipsi_non = np.mean(h[j : j + 5, ipsi_non_idx, i], axis=0)
            h_contra_non = np.mean(h[j : j + 5, contra_non_idx, i], axis=0)

            ipsi_ROC[j, i] = rocN(h_ipsi_pref, h_ipsi_non)
            contra_ROC[j, i] = rocN(h_contra_pref, h_contra_non)
    return ipsi_ROC, contra_ROC


def plot_all_avg_ROC(H_ROC, M_ROC, L_ROC, Z_ROC, mode, cell_idx=None, save_plt=True):
    H_line = np.nanmean(H_ROC, axis=(0, 2))
    M_line = np.nanmean(M_ROC, axis=(0, 2))
    L_line = np.nanmean(L_ROC, axis=(0, 2))
    Z_line = np.nanmean(Z_ROC, axis=(0, 2))

    # calculate STE
    if total_rep > 1:
        H_ste = sem(np.nanmean(H_ROC, 2), nan_policy="omit")
        M_ste = sem(np.nanmean(M_ROC, 2), nan_policy="omit")
        L_ste = sem(np.nanmean(L_ROC, 2), nan_policy="omit")
        Z_ste = sem(np.nanmean(Z_ROC, 2), nan_policy="omit")
    else:
        H_ste = sem(H_ROC[0, :, :], axis=1, nan_policy="omit")
        M_ste = sem(M_ROC[0, :, :], axis=1, nan_policy="omit")
        L_ste = sem(L_ROC[0, :, :], axis=1, nan_policy="omit")
        Z_ste = sem(Z_ROC[0, :, :], axis=1, nan_policy="omit")

    if mode == "sac":
        # perform t-test for sacade selectivity
        if total_rep == 1:
            H_mean = np.nanmean(H_ROC[0, -10:, :], axis=1)
            M_mean = np.nanmean(M_ROC[0, -10:, :], axis=1)
            L_mean = np.nanmean(L_ROC[0, -10:, :], axis=1)
            Z_mean = np.nanmean(Z_ROC[0, -10:, :], axis=1)
        else:
            H_mean = np.nanmean(H_ROC[:, -10:, :], axis=(1, 2))
            M_mean = np.nanmean(M_ROC[:, -10:, :], axis=(1, 2))
            L_mean = np.nanmean(L_ROC[:, -10:, :], axis=(1, 2))
            Z_mean = np.nanmean(Z_ROC[:, -10:, :], axis=(1, 2))
        print("One sample t-test result for saccade selectivity:")
        print("H:")
        print(ttest_1samp(H_mean, 0.5))
        print("M:")
        print(ttest_1samp(M_mean, 0.5))
        print("L:")
        print(ttest_1samp(L_mean, 0.5))
        print("Z:")
        print(ttest_1samp(Z_mean, 0.5))

    fig, ax = plt.subplots()
    ax.plot(H_line, label="H", color="r")
    ax.plot(M_line, label="M", color="g")
    ax.plot(L_line, label="L", color="b")
    ax.plot(Z_line, label="Z", color="k")
    ax.legend(loc="best", prop={"size": 10}, frameon=False)

    ax.fill_between(
        np.arange(h_len), H_line - H_ste, H_line + H_ste, color="r", alpha=0.3
    )
    ax.fill_between(
        np.arange(h_len), M_line - M_ste, M_line + M_ste, color="g", alpha=0.3
    )
    ax.fill_between(
        np.arange(h_len), L_line - L_ste, L_line + L_ste, color="b", alpha=0.3
    )
    ax.fill_between(
        np.arange(h_len), Z_line - Z_ste, Z_line + Z_ste, color="k", alpha=0.3
    )

    ax.set_xlim(0, h_len)
    xticks = np.array([0, round(h_len / 2), h_len])
    ax.set_xticks(xticks)
    ax.set_xticklabels([-500, 0, 500])
    # ax.set_xticklabels((xticks+20-stim_st_time)*20)
    ax.set_ylabel("Average AUC")
    ax.set_xlabel("Time")
    ax.axvline(x=round(h_len / 2), color="k", alpha=0.8, linestyle="--", linewidth=1)
    ax.axvline(
        x=round(h_len / 2) - (stim_st_time - target_st_time),
        color="k",
        alpha=0.8,
        linestyle="--",
        linewidth=1,
    )
    plt.tight_layout()

    if save_plt:
        pic_dir = os.path.join(f_dir, "ROC_plots_%dnet" % total_rep)
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "all_ROC_%s.png" % mode))
        plt.savefig(os.path.join(pic_dir, "all_ROC_%s.pdf" % mode), format="pdf")
        plt.savefig(os.path.join(pic_dir, "all_ROC_%s.eps" % mode), format="eps")
        plt.close(fig)


def plot_all_avg_ROC_sep_sac(line_dict, save_plt=True):
    fig, ax = plt.subplots()
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax.plot(
        np.nanmean(line_dict["H_ipsi"], axis=1), label="H", color="r", linestyle="--"
    )
    ax.plot(
        np.nanmean(line_dict["M_ipsi"], axis=1), label="M", color="g", linestyle="--"
    )
    ax.plot(np.nanmean(line_dict["H_contra"], axis=1), label="H", color="r")
    ax.plot(np.nanmean(line_dict["M_contra"], axis=1), label="M", color="g")
    h_pval = ttest_rel(
        line_dict["H_ipsi"], line_dict["H_contra"], axis=1, nan_policy="omit"
    ).pvalue
    m_pval = ttest_rel(
        line_dict["M_ipsi"], line_dict["M_contra"], axis=1, nan_policy="omit"
    ).pvalue

    ax.plot(
        np.nanmean(line_dict["L_ipsi"], axis=1), label="L", color="b", linestyle="--"
    )
    ax.plot(
        np.nanmean(line_dict["Z_ipsi"], axis=1), label="Z", color="k", linestyle="--"
    )
    ax.plot(np.nanmean(line_dict["L_contra"], axis=1), label="L", color="b")
    ax.plot(np.nanmean(line_dict["Z_contra"], axis=1), label="Z", color="k")
    l_pval = ttest_rel(
        line_dict["L_ipsi"], line_dict["L_contra"], axis=1, nan_policy="omit"
    ).pvalue
    z_pval = ttest_rel(
        line_dict["Z_ipsi"], line_dict["Z_contra"], axis=1, nan_policy="omit"
    ).pvalue
    # ax2.legend(loc='best', prop={'size': 10}, frameon=False)

    # perform ANOVA
    if total_rep == 1:
        H_mean_ipsi = np.mean(line_dict["H_ipsi"][-20:, :], axis=0)
        M_mean_ipsi = np.mean(line_dict["M_ipsi"][-20:, :], axis=0)
        L_mean_ipsi = np.mean(line_dict["L_ipsi"][-20:, :], axis=0)

        H_mean_contra = np.mean(line_dict["H_contra"][-20:, :], axis=0)
        M_mean_contra = np.mean(line_dict["M_contra"][-20:, :], axis=0)
        L_mean_contra = np.mean(line_dict["L_contra"][-20:, :], axis=0)

        coh_arr = ["H", "M", "L"] * 2
        sac_arr = ["ipsi"] * 3 + ["contra"] * 3
        roc_arr = [
            H_mean_ipsi,
            M_mean_ipsi,
            L_mean_ipsi,
            H_mean_contra,
            M_mean_contra,
            L_mean_contra,
        ]
        df = pd.DataFrame({"coh": coh_arr, "sac": sac_arr, "roc": roc_arr})
        df = df.explode("roc")
        df = df[~pd.isnull(df["roc"])]
        df["roc"] = pd.to_numeric(df["roc"])
        model = ols(
            "roc ~ C(sac) + C(coh) +\
            C(sac):C(coh)",
            data=df,
        ).fit()
        twoway_result = sm.stats.anova_lm(model, type=2)
        print("\n")
        print("Saccade selectivity two way ANOVA result:")
        print(twoway_result)

    h_pval_x = np.where(h_pval <= 0.005)[0]
    m_pval_x = np.where(m_pval <= 0.005)[0]
    l_pval_x = np.where(l_pval <= 0.005)[0]
    z_pval_x = np.where(z_pval <= 0.005)[0]

    pval_y1 = max(np.nanmean(line_dict["H_ipsi"], axis=1)) + 0.11
    pval_y2 = pval_y1 - 0.005
    pval_y3 = pval_y2 - 0.005
    pval_y4 = pval_y3 - 0.005

    xticks = np.array([0, round(h_len / 2), h_len])

    ax.set_xlim(0, h_len)
    ax.set_xticks(xticks)
    ax.set_xticklabels([-500, 0, 500])
    ax.set_ylabel("Average AUC")
    ax.set_xlabel("Time")
    ax.axvline(x=round(h_len / 2), color="k", alpha=0.8, linestyle="--", linewidth=1)
    ax.axvline(
        x=round(h_len / 2) - (stim_st_time - target_st_time),
        color="k",
        alpha=0.8,
        linestyle="--",
        linewidth=1,
    )
    ax.scatter(
        h_pval_x, np.ones(h_pval_x.shape) * pval_y1, color="r", marker="*", linewidths=2
    )
    ax.scatter(
        m_pval_x, np.ones(m_pval_x.shape) * pval_y2, color="g", marker="*", linewidths=2
    )

    ax.set_xlim(0, h_len)
    ax.set_xticks(xticks)
    ax.set_xticklabels([-500, 0, 500])
    ax.set_ylabel("Average AUC")
    ax.set_xlabel("Time")
    ax.axvline(x=round(h_len / 2), color="k", alpha=0.8, linestyle="--", linewidth=1)
    ax.axvline(
        x=round(h_len / 2) - (stim_st_time - target_st_time),
        color="k",
        alpha=0.8,
        linestyle="--",
        linewidth=1,
    )
    ax.scatter(
        l_pval_x, np.ones(l_pval_x.shape) * pval_y3, color="b", marker="*", linewidths=2
    )
    ax.scatter(
        z_pval_x, np.ones(z_pval_x.shape) * pval_y4, color="k", marker="*", linewidths=2
    )

    plt.tight_layout()

    if save_plt:
        pic_dir = os.path.join(f_dir, "ROC_plots_%dnet" % total_rep)
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "all_ROC_dir_sep_sac.png"))
        plt.savefig(os.path.join(pic_dir, "all_ROC_dir_sep_sac.pdf"), format="pdf")
        plt.savefig(os.path.join(pic_dir, "all_ROC_dir_sep_sac.eps"), format="eps")
        plt.close(fig)


if __name__ == "__main__":
    main()
