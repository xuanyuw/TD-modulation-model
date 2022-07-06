
from calc_params import par
from math import ceil
import numpy as np
import brainpy.math as bm
from jax import checking_leaks
from os.path import join
from utils import get_module_idx, get_diff_stim, calc_input_sum
from time import time


def fill_rand_conn(mask, from_rng, to_rng, conn_prob):
    """
    Generates a random mask block of certain size and connection probability
    input: 
        from_rng: source neurons of the projection (start, end)
        to_rng: target neurons of the projection (start, end)
        conn_prob: float,  the connection probability of this block
    """
    sz = (from_rng[1] - from_rng[0], to_rng[1] - to_rng[0])
    mask[from_rng[0]:from_rng[1], to_rng[0]:to_rng[1]] = np.random.choice(
        [0, 1], size=sz, p=(1-conn_prob, conn_prob))
    return mask


def calculate_rf_rngs():
    """Generates the bounds for rf blocks"""
    ei = [par['exc_inh_prop'], 1-par['exc_inh_prop']]
    rf_bnd = np.append(0, np.cumsum(
        [ceil(par['n_hidden'] * eix * p) for eix in ei for p in par['RF_perc']]))
    rf_rngs = [(rf_bnd[n], rf_bnd[n+1]) for n in range(len(rf_bnd)-1)]
    return rf_rngs


def fill_mask(rf_rngs, conn_probs, mask):
    for i in range(len(rf_rngs)):
        for j in range(len(rf_rngs)):
            mask = fill_rand_conn(mask, rf_rngs[i], rf_rngs[j], conn_probs[i, j])
    return mask


def generate_rnn_mask():
    rnn_mask_init = np.zeros((par['n_hidden'], par['n_hidden']))
    h_prob, m_prob, l_prob = par['within_rf_conn_prob'], par['cross_rf_conn_prob'], par['cross_module_conn_prob']
    temp_probs = np.array([[h_prob, m_prob, l_prob, l_prob],
                           [m_prob, h_prob, l_prob, l_prob],
                           [l_prob, l_prob, h_prob, m_prob],
                           [l_prob, l_prob, m_prob, h_prob]])
    # temp_probs = np.array([[h_prob, h_prob, m_prob, m_prob],
    #                        [h_prob, h_prob, m_prob, m_prob],
    #                        [m_prob, m_prob, h_prob, h_prob],
    #                        [m_prob, m_prob, h_prob, h_prob]])
    conn_probs = np.tile(temp_probs, (2, 2))
    rf_rngs = calculate_rf_rngs()
    rnn_mask_init = fill_mask(rf_rngs, conn_probs, rnn_mask_init)
    # remove self_connections
    temp_mask = bm.ones((par['n_hidden'], par['n_hidden'])) - bm.eye(par['n_hidden'])
    rnn_mask_init = rnn_mask_init * temp_mask
    return rnn_mask_init

def get_fix_conn_mask(in_rng, rf_rng, conn_prob):
    sz = (in_rng[1] - in_rng[0], rf_rng[1] - rf_rng[0])
    temp = np.zeros(sz).flatten()
    conn_idx = np.random.choice(np.arange(len(temp)), int(len(temp)*conn_prob), replace=False)
    temp[conn_idx] = 1
    temp_mask = temp.reshape(sz)
    return temp_mask

def generate_in_mask():
    in_mask_init = np.zeros((par['n_input'], par['n_hidden']))
    rf_rngs = calculate_rf_rngs()
    in_rngs = [(0, par['num_motion_tuned']), (par['num_motion_tuned'], par['num_motion_tuned'] +
                                              par['num_color_tuned']//2), (par['n_input'] -
                                                                           par['num_color_tuned']//2, par['n_input'])]
    n = par['num_receptive_fields']
    in_idx = par['input_idx']
    for i in range(len(in_idx)):
        # exc conn
        
        temp_mask = get_fix_conn_mask(in_rngs[in_idx[i]], rf_rngs[i], par['input_conn_prob'])
        in_mask_init[in_rngs[in_idx[i]][0]:in_rngs[in_idx[i]][1], rf_rngs[i][0]:rf_rngs[i][1]] = temp_mask
        # in_mask_init[in_rngs[in_idx[i]][0]:in_rngs[in_idx[i]][1], rf_rngs[i][0]:rf_rngs[i][1]] = np.random.choice(
        #     [0, 1], size=sz, p=(1-par['input_conn_prob'], par['input_conn_prob']))
        # # inh conn
        temp_mask = get_fix_conn_mask(in_rngs[in_idx[i]], rf_rngs[i+n], par['input_conn_prob'])
        in_mask_init[in_rngs[in_idx[i]][0]:in_rngs[in_idx[i]][1], rf_rngs[i+n][0]:rf_rngs[i+n][1]] = temp_mask
        # in_mask_init[in_rngs[in_idx[i]][0]:in_rngs[in_idx[i]][1], rf_rngs[i+n][0]:rf_rngs[i+n][1]] = np.random.choice(
        #     [0, 1], size=sz, p=(1-par['input_conn_prob'], par['input_conn_prob']))
    return in_mask_init


def generate_out_mask():
    out_mask_init = np.zeros((par['n_hidden'], par['n_output']))
    rf_rngs = calculate_rf_rngs()
    for idx in range(par['n_output']):
        i = par['output_rf'][idx]
        temp_mask = get_fix_conn_mask(rf_rngs[i], (0, 1), par['output_conn_prob'])
        out_mask_init[rf_rngs[i][0]:rf_rngs[i][1], idx] = temp_mask.flatten()
        # out_mask_init[rf_rngs[i][0]:rf_rngs[i][1], idx] = np.random.choice(
        #     [0, 1], size=(rf_rngs[i][1] - rf_rngs[i][0], ), p=(1-par['output_conn_prob'], par['output_conn_prob']))
    return out_mask_init


def initialize(gamma_shape, size):
    return np.random.gamma(gamma_shape, size=size).astype(np.float32)

def re_init_win(in_weight, in_mask, stim):
    all_module_idx = get_module_idx()
    g_motion, r_motion, m1_g, m1_r = get_diff_stim(stim.generate_trial())
    motion_rf_idx = [0, 2, 4, 6]
    re_init_cond = lambda x: min(x) < 0.8*max(x)
    re_init = True
    print('re-initializing input weights...')
    start = time()
    while re_init:
        g_motion_vals = calc_input_sum(
            in_weight, in_mask, g_motion, [all_module_idx[x] for x in motion_rf_idx]
        )
        re_init = re_init_cond(g_motion_vals)

        r_motion_vals = calc_input_sum(
            in_weight, in_mask, r_motion, [all_module_idx[x] for x in motion_rf_idx]
        )
        re_init = re_init or re_init_cond(r_motion_vals)

        m1_g_vals = calc_input_sum(
            in_weight,
            in_mask,
            m1_g,
            [
                all_module_idx[x]
                for x in range(len(all_module_idx))
                if x not in motion_rf_idx
            ],
        )
        re_init = re_init or re_init_cond(m1_g_vals)

        m1_r_vals = calc_input_sum(
            in_weight,
            in_mask,
            m1_r,
            [
                all_module_idx[x]
                for x in range(len(all_module_idx))
                if x not in motion_rf_idx
            ],
        )
        re_init = re_init or re_init_cond(m1_r_vals)

        if re_init:
            in_weight = initialize(0.1, (par['n_input'], par['n_hidden']))
    end = time()
    print('elapsed time: %f' %(end-start))
    return in_weight

def re_init_wout(out_weight, out_mask):
    masked_weight = out_weight * out_mask
    all_sums = (round(np.sum(masked_weight[:, 0]).astype("float"), 3)), round(np.sum(masked_weight[:, 1]).astype("float"), 3)
    print('re-initializing output weights...')
    start = time()
    while min(all_sums)<0.8*max(all_sums):
        out_weight = initialize(0.1, (par['n_hidden'], par['n_output']))
        masked_weight = out_weight * out_mask
        all_sums = (round(np.sum(masked_weight[:, 0]).astype("float"), 3)), round(np.sum(masked_weight[:, 1]).astype("float"), 3)
    end = time()
    print('elapsed time: %f' %(end-start))
    return out_weight
    



def generate_raw_weights():
    """
    Initialize the weights without multiplying masks 
    The masks will be applied later.
    """
    w_in0 = initialize(0.1, (par['n_input'], par['n_hidden']))
    # w_in0 =  np.random.uniform(0, 0.2, size=(par['n_input'], par['n_hidden']))
    w_rnn0 = initialize(0.2, (par['n_hidden'], par['n_hidden']))
    w_rnn0[:, par['ind_inh']] = initialize(0.2, (par['n_hidden'], len(par['ind_inh'])))
    w_rnn0[par['ind_inh'], :] = initialize(0.2, (len(par['ind_inh']), par['n_hidden']))
    w_rnn0 =  par['EI_matrix'] @  bm.relu(w_rnn0)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] == 'none':
        w_rnn0 = w_rnn0/3.
    w_out0 = initialize(0.1, (par['n_hidden'], par['n_output']))
    # w_out0 = np.random.uniform(0, 0.2, size=(par['n_hidden'], par['n_output']))
    b_rnn0 = np.zeros((1, par['n_hidden']), dtype=np.float32)
    b_out0 = np.zeros((1, par['n_output']), dtype=np.float32)
    return w_in0, w_rnn0, w_out0, b_rnn0, b_out0


def initialize_weights(lr=0, rep=0, stim=None):
    print('Initializing Weights...')
    in_mask_init = generate_in_mask()
    rnn_mask_init = generate_rnn_mask()
    out_mask_init = generate_out_mask()
    w_in0, w_rnn0, w_out0, b_rnn0, b_out0 = generate_raw_weights()
    if stim is not None:
        w_in0 = re_init_win(w_in0, in_mask_init, stim)
        w_out0 = re_init_wout(w_out0, out_mask_init)

    w_in0 *= in_mask_init
    w_rnn0 *= rnn_mask_init
    w_out0 *= out_mask_init

    all_weights = {'in_mask_init': in_mask_init,
                   'rnn_mask_init': rnn_mask_init,
                   'out_mask_init': out_mask_init,
                   'w_in0': w_in0,
                   'w_rnn0': w_rnn0,
                   'w_out0': w_out0,
                   'b_rnn0': b_rnn0,
                   'b_out0': b_out0}
    with open(join(par['save_dir'], 'init_weight_%d_lr%f.pth' % (rep, lr)), 'wb') as f:
        np.save(f, all_weights)
    return all_weights

 