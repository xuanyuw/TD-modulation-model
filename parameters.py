import numpy as np
import json
from itertools import combinations, permutations
from math import ceil

with open('./params.json', 'r') as par_file:
    par = json.load(par_file)


def calc_parameters():
    """Calculate parameters"""
    trial_length = par['time_fixation'] + \
        par['time_target']+par['time_stim']
    # Length of each trial in time steps
    par['num_time_steps'] = trial_length//par['dt']
    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + \
        par['num_fix_tuned'] + par['num_color_tuned']
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    par['fix_time_rng'] = np.arange(par['time_fixation']//par['dt'])
    par['target_time_rng'] = np.arange(
        par['time_fixation']//par['dt'], (par['time_fixation']+par['time_target'])//par['dt'])
    par['stim_time_rng'] = np.arange(
        (par['time_fixation']+par['time_target'])//par['dt'], trial_length//par['dt'])
    par['n_hidden'] = np.sum(par['module_sizes'])

    EI_list = np.ones([par['n_hidden']], dtype=np.float32)
    module_bounds = np.cumsum(par['module_sizes'])
    inh_bounds = module_bounds - \
        np.array(par['module_sizes'])*(1-par['exc_inh_prop'])
    ind_inh = np.hstack([np.arange(inh_bounds[i], module_bounds[i])
                         for i in range(len(inh_bounds))]).astype('int')
    par['ind_inh'] = ind_inh
    EI_list[ind_inh] = -1.
    par['EI_list'] = EI_list
    # EI matrix
    par['EI_matrix'] = np.diag(EI_list)
    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt']/(par['membrane_time_constant']))
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    # since term will be multiplied by par['alpha_neuron']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']
    # weight cost in loss function
    par['weight_cost'] = par['lambda_weight'] / np.square(par['n_hidden'])
    # initial neural activity
    par['x0'] = 0.1*np.ones((1, par['n_hidden']), dtype=np.float32)


def update_synaptic_config():

    synaptic_configurations = {
        'full': ['facilitating' if i % 2 == 0 else 'depressing' for i in range(par['n_hidden'])],
        'fac': ['facilitating' for i in range(par['n_hidden'])],
        'dep': ['depressing' for i in range(par['n_hidden'])],
        'exc_fac': ['facilitating' if par['EI_list'][i] == 1 else 'static' for i in range(par['n_hidden'])],
        'exc_dep': ['depressing' if par['EI_list'][i] == 1 else 'static' for i in range(par['n_hidden'])],
        'inh_fac': ['facilitating' if par['EI_list'][i] == -1 else 'static' for i in range(par['n_hidden'])],
        'inh_dep': ['depressing' if par['EI_list'][i] == -1 else 'static' for i in range(par['n_hidden'])],
        'exc_dep_inh_fac': ['depressing' if par['EI_list'][i] == 1 else 'facilitating' for i in range(par['n_hidden'])],
        'none': [None for i in range(par['n_hidden'])]
    }

    # initialize synaptic values
    par['alpha_stf'] = np.ones((1, par['n_hidden']), dtype=np.float32)
    par['alpha_std'] = np.ones((1, par['n_hidden']), dtype=np.float32)
    par['U'] = np.ones((1, par['n_hidden']), dtype=np.float32)
    par['syn_x_init'] = np.ones(
        (par['batch_size'], par['n_hidden']), dtype=np.float32)
    par['syn_u_init'] = 0.3 * \
        np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)
    par['dynamic_synapse'] = np.zeros((1, par['n_hidden']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if synaptic_configurations[par['synapse_config']][i] == 'facilitating':
            par['alpha_stf'][0, i] = par['dt']/par['tau_slow']
            par['alpha_std'][0, i] = par['dt']/par['tau_fast']
            par['U'][0, i] = 0.15
            par['syn_u_init'][:, i] = par['U'][0, i]
            par['dynamic_synapse'][0, i] = 1

        elif synaptic_configurations[par['synapse_config']][i] == 'depressing':
            par['alpha_stf'][0, i] = par['dt']/par['tau_fast']
            par['alpha_std'][0, i] = par['dt']/par['tau_slow']
            par['U'][0, i] = 0.45
            par['syn_u_init'][:, i] = par['U'][0, i]
            par['dynamic_synapse'][0, i] = 1


def update_parameters(updates):
    """ 
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)] ,
    Note: this function does not update weights, if the parameters changed are involved in weight or 
        mask calculation, need to call initialize_weight and update_weight after this function
    """

    # np.random.seed(10)
    for key, val in updates.items():
        par[key] = val
    # re-calculate the params given the new value
    calc_parameters()
    update_synaptic_config()


###########################################################################


def generate_random_mask(from_t, to_t, conn_prob):
    """
    Generates a random mask block of certain size and connection probability
    input: 
        from_t: source neurons of the projection (start, end)
        to_t: target neurons of the projection (start, end)
        conn_prob: float,  the connection probability of this block
    """
    size = (from_t[1] - from_t[0], to_t[1] - to_t[0])
    return np.random.choice([0, 1], size=size, p=(1-conn_prob, conn_prob))


def calculate_rf_rngs():
    """Generates the bounds for rf blocks"""
    ei = [par['exc_inh_prop'], 1-par['exc_inh_prop']]
    m1_bnd = np.append(0, np.cumsum(
        [ceil(par['module_sizes'][0] * eix * p) for eix in ei for p in par['RF_receiver_perc']]))
    m2_bnd = np.append(0, np.cumsum([ceil(par['module_sizes'][-1] * eix * p)
                                     for eix in ei for p in par['RF_projector_perc']]))+sum(par['module_sizes'][:-1])
    # sc_bnd = np.append(0, np.cumsum([ceil(par['module_sizes'][2] * eix * p)
    #                                  for eix in ei for p in par['RF_projector_perc']]))+sum(par['module_sizes'][:1])
    # middle layer doesn't have rf
    md_blk = []
    if len(par['module_sizes']) > 2:
        md_bnd = np.append(0, np.cumsum(
            [ceil(par['module_sizes'][1] * eix) for eix in ei]))+par['module_sizes'][0]
        md_blk = [(md_bnd[n], md_bnd[n+1]) for n in range(len(md_bnd)-1)]

    m1_blk = [(m1_bnd[n], m1_bnd[n+1]) for n in range(len(m1_bnd)-1)]
    m2_blk = [(m2_bnd[n], m2_bnd[n+1]) for n in range(len(m2_bnd)-1)]

    return m1_blk, m2_blk, md_blk


def generate_rf_module_conn(rfs, comb, cross_mod=False):
    """
    Generate same module (cross_mod=False) or cross module connections (cross_mod=True)
    input:
        rfs: each element is ((from_start, from_end), (to_start, to_end)) of a receptive field
        comb: each element is ((from_start, from_end), (to_start, to_end)) of a non-rf section
        cross_mod: calculate same or cross module connections
    output: 
        if cross_mod=True, then output the index of the neurons that receive inputs, else return none.
    """
    if cross_mod:
        self_conn_prob = par['between_module_rf_conn_prob']
        conn_prob = par['between_module_conn_prob']
    else:
        self_conn_prob = par['within_rf_conn_prob']
        conn_prob = par['cross_rf_conn_prob']
    for tup in comb:
        par['conn_mask_init'][tup[0][0]:tup[0][1], tup[1][0]:tup[1][1]
                              ] = generate_random_mask(tup[0], tup[1], conn_prob)
        par['conn_mask_init'][tup[1][0]:tup[1][1], tup[0][0]:tup[0][1]
                              ] = generate_random_mask(tup[1], tup[0], conn_prob)
    for tup in rfs:
        par['conn_mask_init'][tup[0][0]:tup[0][1], tup[1][0]:tup[1][1]
                              ] = generate_random_mask(tup[0], tup[1], self_conn_prob)
        if cross_mod:
            par['conn_mask_init'][tup[1][0]:tup[1][1], tup[0][0]:tup[0][1]
                                  ] = generate_random_mask(tup[1], tup[0], self_conn_prob)


def generate_intersection(md_blk, m1_blk, m2_blk):
    # TODO: only support m1 and m2 has same sizes
    # TODO: simplify the code if the intersection is inhibitory only
    rng = (md_blk[0][0], md_blk[1][1])
    par['conn_mask_init'][:, rng[0]:rng[1]] = generate_random_mask(
        (0, sum(par['module_sizes'])), rng, par['intersec_input_prob'])
    par['conn_mask_init'][rng[0]:rng[1], :] = generate_random_mask(
        rng, (0, sum(par['module_sizes'])), par['intersec_input_prob'])

    temp_rng = (rng[1]-round(par['module_sizes'][1]
                * par['intersec_inh_prop']), rng[1])
    if not par['input_to_intersec_inh']:
        par['conn_mask_init'][:, temp_rng[0]:temp_rng[1]] = 0

    if not par['inh_conn_from_intersec']:
        par['conn_mask_init'][temp_rng[0]:temp_rng[1], :] = 0

    if not par['inh_conn_to_intersec']:
        par['conn_mask_init'][par['ind_inh'], rng[0]:rng[1]] = 0

    if not par['stim_intersec_conn']:
        stim_idx = np.hstack([np.arange(m1_blk[0][0], m1_blk[0][1]), np.arange(
            m1_blk[3][0], m1_blk[3][1]), np.arange(m2_blk[0][0], m2_blk[0][1]), np.arange(m2_blk[3][0], m2_blk[3][1])])
        par['conn_mask_init'][stim_idx, rng[0]:rng[1]] = 0
        par['conn_mask_init'][rng[0]:rng[1], stim_idx] = 0

    # intersection self connections
    par['conn_mask_init'][rng[0]:rng[1], rng[0]:rng[1]
                          ] = generate_random_mask(rng, rng, par['intersec_self_conn_prob'])


def add_interneurons(blk, inter1_idx, inter2_idx):
    t1_exc, t2_exc = blk[1], blk[2]

    # t1 inh to t2 exc
    eff_size_t1 = round(par['interneuron_eff_prop']*(t2_exc[1]-t2_exc[0]))
    eff_size_t2 = round(par['interneuron_eff_prop']*(t1_exc[1]-t1_exc[0]))

    t2_rec_idx = np.random.choice(range(t2_exc[0], t2_exc[1]), size=(
        par['interneuron_num'], eff_size_t1), replace=True)
    for i in range(par['interneuron_num']):
        par['conn_mask_init'][inter1_idx[i], t2_rec_idx[i, :]] = 1
    t1_proj_idx = np.random.choice(range(t1_exc[0], t1_exc[1]), size=(
        par['interneuron_num'], eff_size_t2), replace=True)
    for i in range(par['interneuron_num']):
        par['conn_mask_init'][t1_proj_idx[i, :], inter1_idx[i]] = 1
    # t2 inh to t1 exc
    t1_rec_idx = np.random.choice(range(t1_exc[0], t1_exc[1]), size=(
        par['interneuron_num'], eff_size_t2), replace=True)
    for i in range(par['interneuron_num']):
        par['conn_mask_init'][inter2_idx[i], t1_rec_idx[i, :]] = 1
    t2_proj_idx = np.random.choice(range(t2_exc[0], t2_exc[1]), size=(
        par['interneuron_num'], eff_size_t1), replace=True)
    for i in range(par['interneuron_num']):
        par['conn_mask_init'][t2_proj_idx[i, :], inter2_idx[i]] = 1
    par['conn_mask_init']


def generate_rnn_mask():
    """Generates mask for RNN with RF"""
    par['conn_mask_init'] = np.zeros((par['n_hidden'], par['n_hidden']))
    m1_blk, m2_blk, md_blk = calculate_rf_rngs()

    # same module connection
    m1_comb = list(combinations(m1_blk, 2))
    m1_rfs = [(m1_blk[n], m1_blk[n]) for n in range(len(m1_blk))]
    m2_comb = list(combinations(m2_blk, 2))
    m2_rfs = [(m2_blk[n], m2_blk[n]) for n in range(len(m2_blk))]
    generate_rf_module_conn(m1_rfs, m1_comb)
    generate_rf_module_conn(m2_rfs, m2_comb)

    # different module connections
    # TODO: only suppport m1 and m2 has same number of rfs
    cross_comb = list(combinations(m1_blk+m2_blk, 2))
    cross_rfs = [(m1_blk[n], m2_blk[n])
                 for n in range(len(par['RF_receiver_perc']))]
    generate_rf_module_conn(
        cross_rfs, cross_comb, cross_mod=True)
    # remove m1-->m2 inhibitory connections
    rng = np.array(
        [ceil(par['module_sizes'][0]*par['exc_inh_prop']), par['module_sizes'][0]])
    par['conn_mask_init'][rng[0]:rng[1], m2_blk[0][0]:m2_blk[-1][-1]] = 0
    # remove m2-->m1 inhibitory connections
    rng += sum(par['module_sizes'][:-1])
    par['conn_mask_init'][rng[0]:rng[1], m1_blk[0][0]:m1_blk[-1][-1]] = 0

    # intersection connections
    if len(par['module_sizes']) > 2:
        generate_intersection(md_blk, m1_blk, m2_blk)
    # increase same module exc-inh connection probability
    for i in range(par['num_receptive_fields']):
        m1_to = m1_blk[i]
        m1_from = m1_blk[i+par['num_receptive_fields']]
        m2_to = m2_blk[i]
        m2_from = m2_blk[i+par['num_receptive_fields']]
        par['conn_mask_init'][m1_from[0]:m1_from[1], m1_to[0]:m1_to[1]
                              ] = generate_random_mask(m1_from, m1_to, par['within_rf_conn_prob'])
        par['conn_mask_init'][m2_from[0]:m2_from[1], m2_to[0]:m2_to[1]
                              ] = generate_random_mask(m2_from, m2_to, par['within_rf_conn_prob'])
        par['conn_mask_init'][m1_to[0]:m1_to[1], m1_from[0]:m1_from[1]
                              ] = generate_random_mask(m1_to, m1_from, par['within_rf_conn_prob'])
        par['conn_mask_init'][m2_to[0]:m2_to[1], m2_from[0]:m2_from[1]
                              ] = generate_random_mask(m2_to, m2_from, par['within_rf_conn_prob'])

    if par['interneuron_m1'] or par['interneuron_m2']:
        orig_size = np.sum(par['module_sizes'])
        inter1_idx = range(orig_size, orig_size + par['interneuron_num'])
        inter2_idx = range(
            orig_size + par['interneuron_num'], par['conn_mask_init'].shape[1])
        if par['interneuron_m1']:
            add_interneurons(m1_blk, inter1_idx, inter2_idx)
        if par['interneuron_m2']:
            add_interneurons(m2_blk, inter1_idx, inter2_idx)

    # set diagnal to 0
    np.fill_diagonal(par['conn_mask_init'], 0)
    # remove connections from neurons that receive inputs
    par['fix_mask_t1r'] = par['conn_mask_init'].copy()
    par['fix_mask_t2r'] = par['conn_mask_init'].copy()
    par['conn_mask_t1r'] = par['conn_mask_init'].copy()
    par['conn_mask_t2r'] = par['conn_mask_init'].copy()
    par['fix_mask_t1r'][par['t1r_target_in_idx'], :] = 0
    par['fix_mask_t2r'][par['t2r_target_in_idx'], :] = 0
    par['conn_mask_t1r'][par['t1r_in_idx_m1'], m2_blk[0][0]:m2_blk[-1][-1]] = 0
    par['conn_mask_t2r'][par['t2r_in_idx_m1'], m2_blk[0][0]:m2_blk[-1][-1]] = 0
    par['conn_mask_t1r'][par['t1r_in_idx_m2'], m1_blk[0][0]:m1_blk[-1][-1]] = 0
    par['conn_mask_t2r'][par['t2r_in_idx_m2'], m1_blk[0][0]:m1_blk[-1][-1]] = 0

    par['conn_mask_init'] = par['conn_mask_init']

    return


def random_in_conn(m, rng, self_exc_blk, self_inh_blk, cross_exc_blk=None, cross_inh_blk=None):
    # corresponding rf exc and inh connections
    m[rng[0]:rng[1], self_exc_blk[0]:self_exc_blk[1]] = generate_random_mask(
        rng, self_exc_blk, par['input_conn_prob'])
    m[rng[0]:rng[1], self_inh_blk[0]:self_inh_blk[1]] = generate_random_mask(
        rng, self_inh_blk, par['input_conn_prob'])
    # cross rf exc and inh connection
    if par['cross_target_input']:
        assert (cross_exc_blk is not None) & (cross_inh_blk is not None)
        m[rng[0]:rng[1], cross_exc_blk[0]:cross_exc_blk[1]] = generate_random_mask(
            rng, cross_exc_blk, par['cross_target_input_prob'])
        m[rng[0]:rng[1], cross_inh_blk[0]:cross_inh_blk[1]] = generate_random_mask(
            rng, cross_inh_blk, par['cross_target_input_prob'])


def divide_input_pool(exc_pool, inh_pool):
    t1r_idx_exc = np.random.choice(
        exc_pool, size=round(len(exc_pool)/2), replace=False)
    t2r_idx_exc = np.setdiff1d(exc_pool, t1r_idx_exc)
    t1r_idx_inh = np.random.choice(
        inh_pool, size=round(len(inh_pool)/2), replace=False)
    t2r_idx_inh = np.setdiff1d(inh_pool, t1r_idx_inh)
    return np.append(t1r_idx_exc, t1r_idx_inh), np.append(t2r_idx_exc, t2r_idx_inh)


def arrange_color_input(blk):
    """blk: the blk of the module that receives major color input"""

    t1_major_exc_pool = np.random.choice(range(blk[1][0], blk[1][1]), size=round(
        par['input_conn_prob']*(blk[1][1]-blk[1][0])), replace=False)
    t1_major_inh_pool = np.random.choice(range(
        blk[-2][0], blk[-2][1]), size=round(par['input_conn_prob']*(blk[-2][1]-blk[-2][0])), replace=False)
    t2_major_exc_pool = np.random.choice(range(blk[2][0], blk[2][1]), size=round(
        par['input_conn_prob']*(blk[2][1]-blk[2][0])), replace=False)
    t2_major_inh_pool = np.random.choice(range(
        blk[-1][0], blk[-1][1]), size=round(par['input_conn_prob']*(blk[-1][1]-blk[-1][0])), replace=False)

    coef = -1 if min(t1_major_exc_pool) > par['module_sizes'][0] else 1
    shift = par['module_sizes'][0] + par['module_sizes'][1] if len(
        par['module_sizes']) > 2 else par['module_sizes'][0]

    t1_minor_exc_pool = np.random.choice(t1_major_exc_pool, size=round(
        par['color_in_minor_perc']*len(t1_major_exc_pool)), replace=False) + coef*shift
    t1_minor_inh_pool = np.random.choice(t1_major_inh_pool, size=round(
        par['color_in_minor_perc']*len(t1_major_inh_pool)), replace=False) + coef*shift
    t2_minor_exc_pool = np.random.choice(t2_major_exc_pool, size=round(
        par['color_in_minor_perc']*len(t2_major_exc_pool)), replace=False) + coef*shift
    t2_minor_inh_pool = np.random.choice(t2_major_inh_pool, size=round(
        par['color_in_minor_perc']*len(t2_major_inh_pool)), replace=False) + coef*shift

    t1r_t1_major, t2r_t1_major = divide_input_pool(
        t1_major_exc_pool, t1_major_inh_pool)
    t1r_t2_major, t2r_t2_major = divide_input_pool(
        t2_major_exc_pool, t2_major_inh_pool)
    t1r_t1_minor, t2r_t1_minor = divide_input_pool(
        t1_minor_exc_pool, t1_minor_inh_pool)
    t1r_t2_minor, t2r_t2_minor = divide_input_pool(
        t2_minor_exc_pool, t2_minor_inh_pool)

    return t1r_t1_major, t2r_t1_major, t1r_t2_major, t2r_t2_major, t1r_t1_minor, t2r_t1_minor, t1r_t2_minor, t2r_t2_minor


def generate_in_mask():
    """Generates mask for input"""
    w_in_mask = np.zeros((par['n_input'], par['n_hidden']))
    # motion stimulus input of m1 and m2
    m1_blk, m2_blk, _ = calculate_rf_rngs()
    stim_in_rng_m1 = (m1_blk[0][0], m1_blk[0][0] +
                      ceil((m1_blk[0][1]-m1_blk[0][0])*par['input_conn_prob']*(1-par['color_in_minor_perc'])))
    stim_in_rng_m2 = (m2_blk[0][0], m2_blk[0][0] +
                      ceil((m2_blk[0][1]-m2_blk[0][0])*par['input_conn_prob']*par['color_in_minor_perc']))
    w_in_mask[:par['num_motion_tuned'],
              stim_in_rng_m1[0]:stim_in_rng_m1[1]] = 1
    w_in_mask[:par['num_motion_tuned'],
              stim_in_rng_m2[0]:stim_in_rng_m2[1]] = 1
    # target stimulus input, different receiver neurons for different target
    par['w_in_mask_t1r'] = w_in_mask.copy()
    par['w_in_mask_t2r'] = w_in_mask.copy()
    par['w_in_mask_init'] = w_in_mask.copy()
    num_rf_t_neurons = par['num_color_tuned']//(par['num_receptive_fields']-1)

    rng_in1 = (par['num_motion_tuned'],
               par['num_motion_tuned']+num_rf_t_neurons)
    rng_in2 = (par['num_motion_tuned']+num_rf_t_neurons, par['n_input'])

    # randomize the rnn neurons that can receive input for each mask
    t1r_t1_major, t2r_t1_major, t1r_t2_major, t2r_t2_major, t1r_t1_minor, t2r_t1_minor, t1r_t2_minor, t2r_t2_minor = arrange_color_input(
        m2_blk)

    par['w_in_mask_t1r'][rng_in1[0]:rng_in1[1],
                         np.hstack([t1r_t1_major, t1r_t1_minor])] = 1
    par['w_in_mask_t2r'][rng_in1[0]:rng_in1[1],
                         np.hstack([t2r_t1_major, t2r_t1_minor])] = 1
    par['w_in_mask_t1r'][rng_in2[0]:rng_in2[1],
                         np.hstack([t1r_t2_major, t1r_t2_minor])] = 1
    par['w_in_mask_t2r'][rng_in2[0]:rng_in2[1],
                         np.hstack([t2r_t2_major, t2r_t2_minor])] = 1

    par['w_in_mask_init'][rng_in1[0]:rng_in1[1], np.hstack(
        [t1r_t1_major, t1r_t1_minor, t1r_t2_major, t1r_t2_minor])] = 1
    par['w_in_mask_init'][rng_in1[0]:rng_in1[1], np.hstack(
        [t2r_t1_major, t2r_t1_minor, t2r_t2_major, t2r_t2_minor])] = 1

    par['t1r_in_idx_m1'] = np.unique(np.hstack(
        [range(stim_in_rng_m1[0], stim_in_rng_m1[1]), t1r_t1_minor, t1r_t2_minor]))
    par['t2r_in_idx_m1'] = np.unique(np.hstack(
        [range(stim_in_rng_m1[0], stim_in_rng_m1[1]), t2r_t1_minor, t2r_t2_minor]))
    par['t1r_in_idx_m2'] = np.unique(np.hstack(
        [range(stim_in_rng_m2[0], stim_in_rng_m2[1]), t1r_t1_major, t1r_t2_major]))
    par['t2r_in_idx_m2'] = np.unique(np.hstack(
        [range(stim_in_rng_m2[0], stim_in_rng_m2[1]), t2r_t1_major, t2r_t2_major]))
    par['t1r_target_in_idx'] = np.unique(
        np.hstack([t1r_t1_minor, t1r_t2_minor, t1r_t1_major, t1r_t2_major]))
    par['t2r_target_in_idx'] = np.unique(
        np.hstack([t2r_t1_minor, t2r_t2_minor, t2r_t1_major, t2r_t2_major]))


def generate_out_mask():
    """Generates mask for output, now only support two outputs"""
    w_out_mask_init = np.zeros((par['n_hidden'], par['n_output']))
    _, m2_blk, _ = calculate_rf_rngs()
    if par['stim_proj'] > 0:
        stim_proj_idx = np.random.randint(m2_blk[0][0], m2_blk[0][1], round(
            (m2_blk[0][1]-m2_blk[0][0])*par['stim_proj']))
        w_out_mask_init[stim_proj_idx, :] = 1

    t1_proj_idx = np.random.randint(m2_blk[1][0], m2_blk[1][1], round(
        (m2_blk[1][1]-m2_blk[1][0])*par['output_conn_prob']))

    t2_proj_idx = np.random.randint(m2_blk[2][0], m2_blk[2][1], round(
        (m2_blk[2][1]-m2_blk[2][0])*par['output_conn_prob']))

    t1_proj_idx = np.append(t1_proj_idx, np.random.choice(t1_proj_idx, size=round(
        par['m1_output_perc']*len(t1_proj_idx)), replace=False)-par['module_sizes'][0])
    t2_proj_idx = np.append(t2_proj_idx, np.random.choice(t2_proj_idx, size=round(
        par['m1_output_perc']*len(t2_proj_idx)), replace=False)-par['module_sizes'][0])

    w_out_mask_init[t1_proj_idx, :round(par['n_output']/2)] = 1
    w_out_mask_init[t2_proj_idx, round(par['n_output']/2):] = 1

    if par['cross_target_output_prob'] > 0:
        t1_x_idx = np.random.randint(m2_blk[1][0], m2_blk[1][1], round(
            (m2_blk[1][1]-m2_blk[1][0])*par['cross_target_output_prob']))
        w_out_mask_init[t1_x_idx, round(par['n_output']/2):] = 1
        t2_x_idx = np.random.randint(m2_blk[2][0], m2_blk[2][1], round(
            (m2_blk[2][1]-m2_blk[2][0])*par['cross_target_output_prob']))
        w_out_mask_init[t2_x_idx, :round(par['n_output']/2)] = 1

    # the nerons in m2 that receive input cannot output
    par['w_out_mask_t1r'] = w_out_mask_init.copy()
    par['w_out_mask_t2r'] = w_out_mask_init.copy()
    par['w_out_mask_t1r'][np.intersect1d(range(
        m2_blk[0][0], m2_blk[-1][-1]), np.append(par['t1r_in_idx_m1'], par['t1r_in_idx_m2'])), :] = 0
    par['w_out_mask_t2r'][np.intersect1d(range(
        m2_blk[0][0], m2_blk[-1][-1]), np.append(par['t2r_in_idx_m1'], par['t2r_in_idx_m2'])), :] = 0
    par['w_out_mask_init'] = w_out_mask_init


def initialize_weights():
    """
    Initialize the weights without multiplying masks and EI matrix
    The masks will be applied later.
    """
    generate_in_mask()
    generate_rnn_mask()
    generate_out_mask()

    par['w_in0'] = np.random.gamma(
        size=[par['n_input'], par['n_hidden']], shape=0.1, scale=1.).astype('float32')
    par['w_rnn_base0'] = np.random.gamma(
        size=(par['n_hidden'], par['n_hidden']), scale=1., shape=0.1).astype('float32')
    par['w_rnn_base0'][:, par['ind_inh']] = np.random.gamma(
        size=(par['n_hidden'], len(par['ind_inh'])), scale=1., shape=0.2)
    par['w_rnn_base0'][par['ind_inh'], :] = np.random.gamma(
        size=(len(par['ind_inh']), par['n_hidden']), scale=1., shape=0.2)
    # set negative weights to 0
    par['w_rnn_base0'][par['w_rnn_base0'] < 0] = 0
    par['w_rnn_base0'] = np.multiply(par['w_rnn_base0'], par['conn_mask_init'])

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] is 'none':
        par['w_rnn_base0'] = par['w_rnn_base0']/3.

    par['w_out0'] = np.random.gamma(
        size=[par['n_hidden'], par['n_output']], shape=0.1, scale=1.).astype(np.float32)
    par['b_rnn0'] = np.zeros((1, par['n_hidden']), dtype=np.float32)
    par['b_out0'] = np.zeros((1, par['n_output']), dtype=np.float32)
