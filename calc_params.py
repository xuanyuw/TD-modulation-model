import numpy as np
import json

print("Initializing Parameters...")

with open("./params.json", "r") as par_file:
    par = json.load(par_file)


def calc_parameters():
    """Calculate parameters"""
    trial_length = par["time_fixation"] + par["time_target"] + par["time_stim"]
    # Length of each trial in time steps
    par["num_time_steps"] = trial_length // par["dt"]
    # Number of input neurons
    par["n_input"] = (
        par["num_motion_tuned"] + par["num_fix_tuned"] + par["num_color_tuned"]
    )
    # The time step in seconds
    par["dt_sec"] = par["dt"] / 1000
    # Length of each trial in ms
    par["fix_time_rng"] = np.arange(par["time_fixation"] // par["dt"])
    par["target_time_rng"] = np.arange(
        par["time_fixation"] // par["dt"],
        (par["time_fixation"] + par["time_target"]) // par["dt"],
    )
    par["stim_time_rng"] = np.arange(
        (par["time_fixation"] + par["time_target"]) // par["dt"],
        trial_length // par["dt"],
    )
    # EI matrix
    par["n_total"] = par["n_hidden"] + par["n_inter"]
    par["EI_list"] = np.ones([par["n_total"]], dtype=np.float32)
    par["ind_inh"] = np.arange(
        par["n_hidden"] * par["exc_inh_prop"], par["n_total"]
    ).astype(int)

    par["EI_list"][par["ind_inh"]] = -1
    par["EI_matrix"] = np.diag(par["EI_list"])
    # Membrane time constant of RNN neurons
    par["alpha_neuron"] = np.float32(par["dt"] / (par["membrane_time_constant"]))
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par["noise_rnn"] = np.sqrt(2 * par["alpha_neuron"]) * par["noise_rnn_sd"]
    # since term will be multiplied by par['alpha_neuron']
    par["noise_in"] = np.sqrt(2 / par["alpha_neuron"]) * par["noise_in_sd"]
    # weight cost in loss function
    par['weight_cost'] = par['lambda_weight'] / par['n_hidden']**2
    # initial neural activity
    par["x0"] = 0.1 * np.ones((par["batch_size"], par["n_total"]), dtype=np.float32)


def update_synaptic_config():
    # 1 = facilitating, -1 =  depressing, 0 = static
    # synaptic_configurations = {
    #     'full': [1 if i % 2 == 0 else -1 for i in range(par['n_hidden'])],
    #     'fac': [1] * par['n_hidden'],
    #     'dep': [-1] * par['n_hidden'],
    #     'exc_fac': [1 if par['EI_list'][i] == 1 else 0 for i in range(par['n_hidden'])],
    #     'exc_dep': [-1 if par['EI_list'][i] == 1 else 0 for i in range(par['n_hidden'])],
    #     'inh_fac': [1 if par['EI_list'][i] == -1 else 0 for i in range(par['n_hidden'])],
    #     'inh_dep': [-1 if par['EI_list'][i] == -1 else 0 for i in range(par['n_hidden'])],
    #     'exc_dep_inh_fac': [-1 if par['EI_list'][i] == 1 else 1 for i in range(par['n_hidden'])],
    #     'none': [0] * par['n_hidden']
    # }

    # need to be filled if use other configurations
    if par["synapse_config"] == "full":
        synaptic_configurations = [
            1 if i % 2 == 0 else -1 for i in range(par["n_total"])
        ]
    else:
        synaptic_configurations = [0] * par["n_total"]

    # initialize synaptic values d
    fac_idx = np.where(np.array(synaptic_configurations) == 1)[0]
    dep_idx = np.where(np.array(synaptic_configurations) == -1)[0]

    par["alpha_stf"] = np.ones((par["n_total"],), dtype=np.float32)
    par["alpha_stf"][fac_idx] = par["dt"] / par["tau_slow"]
    par["alpha_stf"][dep_idx] = par["dt"] / par["tau_fast"]

    par["alpha_std"] = np.ones((par["n_total"],), dtype=np.float32)
    par["alpha_std"][fac_idx] = par["dt"] / par["tau_fast"]
    par["alpha_std"][dep_idx] = par["dt"] / par["tau_slow"]

    par["U"] = np.ones((par["n_total"],), dtype=np.float32)
    par["U"][fac_idx] = par['U_stf']
    par["U"][dep_idx] = par['U_std']

    par["syn_x_init"] = np.ones((par["batch_size"], par["n_total"]), dtype=np.float32)

    par["syn_u_init"] = 0.3 * np.ones(
        (par["batch_size"], par["n_total"]), dtype=np.float32
    )
    par["syn_u_init"][:, fac_idx] = par["U"][fac_idx]
    par["syn_u_init"][:, dep_idx] = par["U"][dep_idx]

    par["dynamic_synapse"] = np.zeros((par["n_total"],), dtype=np.float32)
    par["dynamic_synapse"][fac_idx] = 1
    par["dynamic_synapse"][dep_idx] = 1


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


calc_parameters()
update_synaptic_config()
