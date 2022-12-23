import numpy as np
from calc_params import par, update_parameters
from model_func import trial
from os import makedirs
from os.path import dirname, exists
import time
from gc import collect

# For debugging
# from jax.config import config
# config.update("jax_debug_nans", True)b


def try_model(par, train):
    try:
        trial(par, train=train)

    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt') 

tic = time.perf_counter()
# Run models
#synaptic_configs = ['full', 'fac', 'dep', 'exc_fac', 'exc_dep', 'inh_fac', 'inh_dep', 'exc_dep_inh_fac', 'none']
synaptic_configs = ['full']
# for rep in np.arange(par['rep'],par['rep_num']):
for rep in range(49, 50):
    for syn_config in synaptic_configs:
        for lr in par['learning_rate_li']: 
            # for shuf_n in range(par['shuffle_num']):
            for shuf_n in range(97, 100):
                update_parameters({'synapse_config': syn_config,
                                'rep': rep,
                                'save_fn': 'model_results_%d_lr%f.pkl' % (rep, lr),
                                'batch_size': par['train_batch_size'],
                                'num_iterations': par['num_train_iterations'],
                                'coherence_levels': par['train_coherence_levels'],
                                'weight_fn': 'weight_%d_lr%f.pth' % (rep, lr),
                                'learning_rate': lr})
                if not exists(dirname(par['save_dir'])):
                    makedirs(dirname(par['save_dir']))
                # print('Training model %d'%rep)
                # try_model(par, True)
                update_parameters({'synapse_config': syn_config,
                                'rep': rep,
                                'shuffle': shuf_n,
                                'save_fn': 'test_results_%d_shuf%d.pkl' %(rep, shuf_n),
                                'batch_size': par['test_batch_size'],
                                'num_iterations': par['num_test_iterations'],
                                'coherence_levels': par['test_coherence_levels']
                                })
                if shuf_n % 20 == 0:
                    print('Testing model %d shuffle # %d'%(rep, shuf_n))
                try_model(par, False)
            collect()
toc = time.perf_counter()

print('elapsed time = %.2f seconds'%(toc-tic))