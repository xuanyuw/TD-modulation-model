import sys
import os

import numpy as np
from parameters import *
from model import *
from model_func import *


def try_model(device):
    try:
        # Run model
        print_important_params()
        train_loop(device)
        test_loop(device)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')


# Wrapped into main
if __name__ == "__main__":

    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        device = torch.device('cuda')
        par['gpu'] = True
    else:
        gpu_id = None
        device = 'cpu'
        par['gpu'] = False

    if not os.path.exists(par['save_dir']):
        os.makedirs(par['save_dir'])

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

                try_model(device)
