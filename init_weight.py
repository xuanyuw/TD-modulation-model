from calc_params import par
from math import ceil
import numpy as np
from os.path import exists
from numpy import load


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
            mask = fill_rand_conn(
                mask, rf_rngs[i], rf_rngs[j], conn_probs[i, j])
    return mask


def generate_rnn_mask():
    rnn_mask_init = np.zeros((par['n_hidden'], par['n_hidden']))
    h_prob, m_prob, l_prob = par['within_rf_conn_prob'], par['cross_rf_conn_prob'], par['cross_module_conn_prob']
    temp_probs = np.array([[h_prob, m_prob, l_prob],
                           [m_prob, h_prob, l_prob],
                           [l_prob, l_prob, h_prob]])
    conn_probs = np.tile(temp_probs, (2, 2))
    rf_rngs = calculate_rf_rngs()
    rnn_mask_init = fill_mask(rf_rngs, conn_probs, rnn_mask_init)
    return rnn_mask_init


def generate_in_mask():
    in_mask_init = np.zeros((par['n_input'], par['n_hidden']))
    rf_rngs = calculate_rf_rngs()
    in_rngs = [(0, par['num_motion_tuned']), (par['num_motion_tuned'], par['num_motion_tuned'] +
                                              par['num_color_tuned']//2), (par['n_input'] -
                                                                           par['num_color_tuned']//2, par['n_input'])]
    n = par['num_receptive_fields']
    for i in range(len(in_rngs)):
        # exc conn
        sz = (in_rngs[i][1] - in_rngs[i][0], rf_rngs[i][1] - rf_rngs[i][0])
        in_mask_init[in_rngs[i][0]:in_rngs[i][1], rf_rngs[i][0]:rf_rngs[i][1]] = np.random.choice(
            [0, 1], size=sz, p=(1-par['input_conn_prob'], par['input_conn_prob']))
        # inh conn
        sz = (in_rngs[i][1] - in_rngs[i][0], rf_rngs[i+n][1] - rf_rngs[i+n][0])
        in_mask_init[in_rngs[i][0]:in_rngs[i][1], rf_rngs[i+n][0]:rf_rngs[i+n][1]] = np.random.choice(
            [0, 1], size=sz, p=(1-par['input_conn_prob'], par['input_conn_prob']))
    return in_mask_init


def generate_out_mask():
    out_mask_init = np.zeros((par['n_hidden'], par['n_output']))
    rf_rngs = calculate_rf_rngs()
    sz = (rf_rngs[2][1] - rf_rngs[1][0], par['n_output'])
    out_mask_init[rf_rngs[1][0]:rf_rngs[2][1], :] = np.random.choice(
        [0, 1], size=sz, p=(1-par['output_conn_prob'], par['output_conn_prob']))
    return out_mask_init


def generate_raw_weights():
    """
    Initialize the weights without multiplying masks and EI matrix
    The masks will be applied later.
    """

    w_in0 = np.random.gamma(
        size=[par['n_input'], par['n_hidden']], shape=0.1, scale=1.).astype('float32')
    w_rnn_base0 = np.random.gamma(
        size=(par['n_hidden'], par['n_hidden']), scale=1., shape=0.1).astype('float32')
    w_rnn_base0[:, par['ind_inh']] = np.random.gamma(
        size=(par['n_hidden'], len(par['ind_inh'])), scale=1., shape=0.2)
    w_rnn_base0[par['ind_inh'], :] = np.random.gamma(
        size=(len(par['ind_inh']), par['n_hidden']), scale=1., shape=0.2)
    # # set negative weights to 0
    # par['w_rnn_base0'][par['w_rnn_base0'] < 0] = 0
    # par['w_rnn_base0'] = np.multiply(par['w_rnn_base0'], par['rnn_mask_init'])

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] is 'none':
        w_rnn_base0 = w_rnn_base0/3.

    w_out0 = np.random.gamma(
        size=[par['n_hidden'], par['n_output']], shape=0.1, scale=1.).astype(np.float32)
    b_rnn0 = np.zeros((1, par['n_hidden']), dtype=np.float32)
    b_out0 = np.zeros((1, par['n_output']), dtype=np.float32)
    return w_in0, w_rnn_base0, w_out0, b_rnn0, b_out0


def initialize_weights():
    in_mask_init = generate_in_mask()
    rnn_mask_init = generate_rnn_mask()
    out_mask_init = generate_out_mask()
    w_in0, w_rnn_base0, w_out0, b_rnn0, b_out0 = generate_raw_weights()

    all_weights = {'in_mask_init': in_mask_init,
                   'rnn_mask_init': rnn_mask_init,
                   'out_mask_init': out_mask_init,
                   'w_in0': w_in0,
                   'w_rnn_base0': w_rnn_base0,
                   'w_out0': w_out0,
                   'b_rnn0': b_rnn0,
                   'b_out0': b_out0}
    with open('weights.npy', 'wb') as f:
        np.save(f, all_weights)
    return all_weights

if exists('weights.npy'):
    with open('weights.npy', 'rb') as f:
        all_weights = load(f, allow_pickle=True)
        all_weights = all_weights.item()
else:
    all_weights = initialize_weights()