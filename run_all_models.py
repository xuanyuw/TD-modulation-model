import numpy as np
from calc_params import par, update_parameters
from model_func import trial


def try_model(par):
    try:
        trial(par)
        trial(par, train=False)  # Run model

    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')


# Run models
#synaptic_configs = ['full', 'fac', 'dep', 'exc_fac', 'exc_dep', 'inh_fac', 'inh_dep', 'exc_dep_inh_fac', 'none']
synaptic_configs = ['full']
for syn_config in synaptic_configs:
    for lr in par['learning_rate_li']:
        for rep in np.arange(par['rep_num']):
            update_parameters({'synapse_config': syn_config,
                               'rep': rep,
                               'save_fn': 'model_results_%s_%d.pkl' % (syn_config, rep),
                               'test_save_fn': 'test_results_%s_%d.pkl' % (syn_config, rep),
                               'weight_fn': 'weight_%s_%d.pth' % (syn_config, rep),
                               'learning_rate': lr})

            print('Synaptic configuration:\t', par['synapse_config'], "\n")

            try_model(par)
