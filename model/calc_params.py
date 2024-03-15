"""
This file contains the parameters for the model. It also contains the function to update the parameters.
"""
import numpy as np

print("Initializing Parameters...")
par = {
    "save_dir": "../test_output_noFeedback_model/", # directory for saving the testing (and training) result
    "model_dir": "full_model/", # directory of the trained model, for ablation testing of the trained model
    "save_fn": "model_results.pkl", # file name for saving model training performance
    "save_train_out": True,
    "save_test_out": True,
    "rep_num": 50, # number of models to train
    "rep": 0, # iteration number of the model, for breakpoints
    "shuffle_num": 0, # number of shuffling weights 
    "shuffle": 0,# iteration number of the shuffle model, for breakpoints
    "cut_spec": [], # can take values of True, False, or []. True: cut selectivity-specific weights, False: cut non-selectivity-specific weights, []: no cutting
    "cut_fb_train": True, # cut feedback weights during training
    "cut_fb_train_factor": 1.176, # make the number of connections roughly the same before and after cutting feedback weights during training

    # Training parameters
    "batch_size": 1024, 
    "train_batch_size": 1024,
    "test_batch_size": 2048,
    "num_iterations": 2000, 
    "num_train_iterations": 200, # Number of maximum iterations for training
    "num_test_iterations": 1,
    "iters_between_outputs": 200, # Print out the training performance every 200 iterations
    "learning_rate": 2e-2,

    # Connection probabilities
    "within_rf_conn_prob": 0.5,
    "cross_rf_conn_prob": 0.25,
    "cross_module_conn_prob": 0.1,
    "input_conn_prob": 0.32,
    "output_conn_prob": 0.32,
    "cross_output_prob": 0.08,

    # Model structure parameters
    "num_motion_tuned": 9,
    "num_color_tuned": 8,
    "n_hidden": 200,
    "n_choices": 2,
    "n_output": 2,
    "exc_inh_prop": 0.8, # proportion of excitatory neurons
    "num_receptive_fields": 4,
    "RF_perc": [0.25, 0.25, 0.25, 0.25], # percentage of neurons in each receptive field
    "input_idx": [0, 1, 0, 2], # input index for each receptive field
    "output_rf": [1, 3], # receptive field index for each output neuron
    
    # STSP parameters
    "dt": 20,
    "membrane_time_constant": 100,
    "tau_fast": 200,
    "tau_dep": 1000,
    "tau_fac": 2000,
    "U_stf": 0.1,
    "U_std": 0.3,

    # Input parameters
    "input_mean": 0.0,
    "noise_in_sd": 0.07,
    "decay_const": 0.33, # stimulus input decaying constant
    "pure_visual_val": 2,
    "motion_mult": 2, # stronger motion signal and noise since motion stimuli are more complex than simple color signals of targets
    "tuning_height": 2, # tuning height of the input neurons
    "kappa": 2, # concentration parameter of the tuning curve
    "coherence_levels": [0, 0.35, 0.55, 0.75, 0.6, 0.9],
    "train_coherence_levels": [0.6, 0.9], # coherence levels for training
    "test_coherence_levels": [0, 0.35, 0.55, 0.75], # coherence levels for testing

    # Weight parameters
    "inout_weight_mean": 0.2,
    "inout_weight_std": 0.05,

    # RNN parameters
    "noise_rnn_sd": 0.08,
    "spike_regularization": "L2",
    "spike_cost": 0.004,
    "clip_max_grad_val": 0.1,
    "lambda_weight": 1,

    # Task parameters
    "time_fixation": 500,
    "time_target": 400,
    "time_stim": 500,
    "time_decision": 100,
    "decision_threshold": 0.8,
}


def calc_parameters():
    """Calculate other parameters"""
    trial_length = par["time_fixation"] + par["time_target"] + par["time_stim"]
    # Length of each trial in time steps
    par["num_time_steps"] = trial_length // par["dt"]
    # Number of input neurons
    par["n_input"] = (
        par["num_motion_tuned"] + par["num_color_tuned"]
    )
    # The time step in seconds
    par["dt_sec"] = par["dt"] / 1000
    # Length of each trial in ms
    # par["fix_time_rng"] = np.arange(par["time_fixation"] // par["dt"])
    par["target_time_rng"] = np.arange(
        par["time_fixation"] // par["dt"],
        (par["time_fixation"] + par["time_target"]) // par["dt"],
    )
    par["stim_time_rng"] = np.arange(
        (par["time_fixation"] + par["time_target"]) // par["dt"],
        trial_length // par["dt"],
    )
    # EI matrix
    par["n_total"] = par["n_hidden"] 
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
    par["weight_cost"] = par["lambda_weight"] / par["n_hidden"] ** 2
    # initial neural activity
    par["x0"] = 0.1 * np.ones((par["batch_size"], par["n_total"]), dtype=np.float32)


def update_synaptic_config():
    # 1 = facilitating, -1 =  depressing, 0 = static
    synaptic_configurations = [
        1 if i % 2 == 0 else -1 for i in range(par["n_total"])
    ]

    # initialize synaptic values d
    fac_idx = np.where(np.array(synaptic_configurations) == 1)[0]
    dep_idx = np.where(np.array(synaptic_configurations) == -1)[0]

    par["alpha_stf"] = np.ones((par["n_total"],), dtype=np.float32)
    par["alpha_stf"][fac_idx] = par["dt"] / par["tau_fac"]
    par["alpha_stf"][dep_idx] = par["dt"] / par["tau_fast"]

    par["alpha_std"] = np.ones((par["n_total"],), dtype=np.float32)
    par["alpha_std"][fac_idx] = par["dt"] / par["tau_fast"]
    par["alpha_std"][dep_idx] = par["dt"] / par["tau_dep"]

    par["U"] = np.ones((par["n_total"],), dtype=np.float32)
    par["U"][fac_idx] = par["U_stf"]
    par["U"][dep_idx] = par["U_std"]

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

    for key, val in updates.items():
        par[key] = val
    # re-calculate the params given the new value
    calc_parameters()
    update_synaptic_config()


calc_parameters()
update_synaptic_config()
