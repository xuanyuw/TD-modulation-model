from init_weight import initialize_weights
from utils import *
from types import SimpleNamespace
import brainpy as bp
import brainpy.math as bm
from numpy import load, tile, save, ndarray
from numpy.random import normal
from os.path import join
from calc_params import par
from tqdm import tqdm
from scipy.stats import ttest_rel

f_dir = (
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
)
total_rep = 50
lr = 0.02


def main():
    spec_conn_num_list = []
    nonspec_conn_num_list = []
    spec_weight_list = []
    nonspec_weight_list = []
    for rep in tqdm(range(total_rep)):
        trained_w = np.load(
            join(f_dir, "weight_%d_lr%f.pth" % (rep, lr)), allow_pickle=True
        )
        trained_w = trained_w.item()
        w_rnn0 = trained_w["w_rnn0"]
        rnn_m = trained_w["rnn_mask_init"]
        spec_loc_exc, spec_loc_inh, nonspec_loc_exc, nonspec_loc_inh = get_conns(
            w_rnn0, rnn_m
        )
        spec_conn_num_list.append(np.sum(spec_loc_exc) + np.sum(spec_loc_inh))
        nonspec_conn_num_list.append(np.sum(nonspec_loc_exc) + np.sum(nonspec_loc_inh))
        spec_weight_list.append(
            np.mean(np.concatenate((w_rnn0[spec_loc_exc], w_rnn0[spec_loc_inh])))
        )
        nonspec_weight_list.append(
            np.mean(np.concatenate((w_rnn0[nonspec_loc_exc], w_rnn0[nonspec_loc_inh])))
        )
    # print(ttest_rel(spec_conn_num_list, nonspec_conn_num_list))
    # print("----------------------")
    # print(ttest_rel(spec_weight_list, nonspec_weight_list))

    plt.figure()
    plt.hist(spec_conn_num_list, bins=10, alpha=0.5, label="spec conn")
    plt.hist(nonspec_conn_num_list, bins=10, alpha=0.5, label="nonspec conn")
    yl = plt.ylim()
    plt.vlines(
        np.mean(spec_conn_num_list),
        0,
        max(yl),
        colors="deepskyblue",
        linestyles="dashed",
        linewidth=2,
    )
    plt.vlines(
        np.mean(nonspec_conn_num_list),
        0,
        max(yl),
        colors="orange",
        linestyles="dashed",
        linewidth=2,
    )
    # plt.hist(np.array(spec_conn_num_list) - np.array(nonspec_conn_num_list))
    plt.legend()
    plt.title("Number of connections")
    plt.savefig(
        os.path.join(
            "F:\Github\TD-modulation-model\generate_figs",
            "connection_comp_num_conn.pdf",
        )
    )

    plt.figure()
    plt.hist(spec_weight_list, bins=10, alpha=0.5, label="spec weight")
    plt.hist(nonspec_weight_list, bins=10, alpha=0.5, label="nonspec weight")
    yl = plt.ylim()
    plt.vlines(
        np.mean(spec_weight_list),
        0,
        max(yl),
        colors="deepskyblue",
        linestyles="dashed",
        linewidth=2,
    )
    plt.vlines(
        np.mean(nonspec_weight_list),
        0,
        max(yl),
        colors="orange",
        linestyles="dashed",
        linewidth=2,
    )
    # plt.hist(np.array(spec_weight_list) - np.array(nonspec_weight_list))
    plt.legend()
    plt.title("Weight values")
    plt.savefig(
        os.path.join(
            "F:\Github\TD-modulation-model\generate_figs",
            "connection_comp_weight_value.pdf",
        )
    )


def get_conns(w_rnn0, rnn_mask):
    # find targ color encoding neurons
    # prefer green target = 0, prefer red target = 1
    m1_targ_rng = np.append(range(40, 80), range(170, 180))
    m2_targ_rng = np.append(range(120, 160), range(190, 200))
    n = SimpleNamespace(
        **load_test_data(
            par["model_dir"],
            "train_output_lr%f_rep%d.h5" % (par["learning_rate"], par["rep"]),
        )
    )
    pref_targ_color = find_pref_targ_color(
        n.h, n.desired_out, n.stim_dir, m1_targ_rng, m2_targ_rng
    )
    m1_green, m1_red, m2_green, m2_red = (
        np.zeros(pref_targ_color.shape),
        np.zeros(pref_targ_color.shape),
        np.zeros(pref_targ_color.shape),
        np.zeros(pref_targ_color.shape),
    )
    m1_green[m1_targ_rng] = pref_targ_color[m1_targ_rng] == 0
    m2_green[m2_targ_rng] = pref_targ_color[m2_targ_rng] == 0
    m1_red[m1_targ_rng] = pref_targ_color[m1_targ_rng] == 1
    m2_red[m2_targ_rng] = pref_targ_color[m2_targ_rng] == 1

    targ_loc = recover_targ_loc(n.desired_out, n.stim_dir)[-1, :]
    normalized_h = min_max_normalize(n.h)
    targ_sel = pick_selective_neurons(
        normalized_h, targ_loc, window_st=25, window_ed=45, alpha=0.05
    )
    m1_green = m1_green * targ_sel
    m2_green = m2_green * targ_sel
    m1_red = m1_red * targ_sel
    m2_red = m2_red * targ_sel

    # find moving direction encoding neurons
    m1_stim_rng = np.append(range(0, 40), range(160, 170))
    m2_stim_rng = np.append(range(80, 120), range(180, 190))
    pref_red = find_pref_dir(n.stim_level, n.stim_dir, n.h)
    pref_green = ~pref_red.astype(bool)
    m1_pref_red, m2_pref_red, m1_pref_green, m2_pref_green = (
        np.zeros(pref_red.shape),
        np.zeros(pref_red.shape),
        np.zeros(pref_red.shape),
        np.zeros(pref_red.shape),
    )
    m1_pref_red[m1_stim_rng] = pref_red[m1_stim_rng]
    m2_pref_red[m2_stim_rng] = pref_red[m2_stim_rng]
    m1_pref_green[m1_stim_rng] = pref_green[m1_stim_rng]
    m2_pref_green[m2_stim_rng] = pref_green[m2_stim_rng]

    stim_sel = pick_selective_neurons(normalized_h, n.stim_dir, alpha=0.05)
    m1_pref_red = m1_pref_red * stim_sel
    m2_pref_red = m2_pref_red * stim_sel
    m1_pref_green = m1_pref_green * stim_sel
    m2_pref_green = m2_pref_green * stim_sel
    # w_rnn =  par['EI_matrix'] @ relu(w_rnn0)

    m1_tr2mr = locate_conn(m1_red, m1_pref_red, rnn_mask)
    m1_tg2mg = locate_conn(m1_green, m1_pref_green, rnn_mask)
    m2_tr2mr = locate_conn(m2_red, m2_pref_red, rnn_mask)
    m2_tg2mg = locate_conn(m2_green, m2_pref_green, rnn_mask)
    match_conn_loc = np.logical_or(
        np.logical_or(m1_tr2mr, m2_tr2mr), np.logical_or(m1_tg2mg, m2_tg2mg)
    )

    m1_tr2mg = locate_conn(m1_red, m1_pref_green, rnn_mask)
    m1_tg2mr = locate_conn(m1_green, m1_pref_red, rnn_mask)
    m2_tr2mg = locate_conn(m2_red, m2_pref_green, rnn_mask)
    m2_tg2mr = locate_conn(m2_green, m2_pref_red, rnn_mask)
    nonmatch_conn_loc = np.logical_or(
        np.logical_or(m1_tr2mg, m2_tr2mg), np.logical_or(m1_tg2mr, m2_tg2mr)
    )

    spec_loc_exc = find_spec_conn(match_conn_loc, w_rnn0, par["EI_matrix"], exc=True)
    spec_loc_inh = find_spec_conn(
        nonmatch_conn_loc, w_rnn0, par["EI_matrix"], exc=False
    )

    nonspec_loc_exc = find_spec_conn(
        match_conn_loc, w_rnn0, par["EI_matrix"], exc=False
    )
    nonspec_loc_inh = find_spec_conn(
        nonmatch_conn_loc, w_rnn0, par["EI_matrix"], exc=True
    )
    return spec_loc_exc, spec_loc_inh, nonspec_loc_exc, nonspec_loc_inh


def find_spec_conn(neu_loc, w_rnn, EI_matrix, exc):
    temp_w_rnn = EI_matrix @ relu(w_rnn)
    if exc:
        sel_loc = (neu_loc != 0) & (temp_w_rnn > 0)
    else:
        sel_loc = (neu_loc != 0) & (temp_w_rnn < 0)
    return sel_loc


if __name__ == "__main__":
    main()
