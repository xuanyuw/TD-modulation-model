"""
Functions used to save model data and to perform analysis
"""

import numpy as np
import jax.numpy as jnp
from os.path import join
from brainpy.math import relu, softmax


def get_perf(target, output, mask, stim_level):
    """ Calculate task accuracy by comparing the actual network output to the desired output
        only examine time points when test stimulus is on, e.g. when y[:,:,0] = 0 """

    target = target.numpy()
    output = output.numpy()
    mask = mask.numpy()

    mask_full = np.float32(mask > 0)
    target_max = np.argmax(target, axis=2)
    output_max = np.argmax(output, axis=2)
    accuracy = {'H': [], 'M': [], 'L': [], 'Z': []}
    #print(target.shape, output.shape, mask.shape)
    #print(target[:,2,:], output[:,2,:])
    for i in range(target_max.shape[1]):
        batch_accuracy = np.sum(np.float32(
            target_max[:, i] == output_max[:, i])*mask_full[:, i])/np.sum(mask_full[:, i])
        if stim_level[i] == 'H':
            accuracy['H'].append(batch_accuracy)
        elif stim_level[i] == 'M':
            accuracy['M'].append(batch_accuracy)
        elif stim_level[i] == 'L':
            accuracy['L'].append(batch_accuracy)
        else:
            accuracy['Z'].append(batch_accuracy)
    accuracy['H'] = np.mean(accuracy['H'])
    accuracy['M'] = np.mean(accuracy['M'])
    accuracy['L'] = np.mean(accuracy['L'])
    accuracy['Z'] = np.mean(accuracy['Z'])
    total_accuracy = np.sum(np.float32(
        target_max == output_max)*mask_full)/np.sum(mask_full)

    return accuracy, total_accuracy


def get_reaction_time(y_output, par):
    start_t = (par['time_fixation'] + par['time_target'])//par['dt']
    #reaciton_time = np.zeros((par['batch_size'], ))
    rt_mask = np.zeros(
        ((par['time_fixation'] + par['time_target'] + par['time_stim'])//par['dt'], 1))
    rt_mask[-par['time_stim']//par['dt']:, :] = 1
    rt_mask = np.repeat(rt_mask, par['batch_size'], axis=1)
    y = softmax(y_output)
    abs_diff = abs(y[:, :, 0] - y[:, :, 1])
    diff = rt_mask * relu(abs_diff - par['decision_threshold'])
    diff_nonzero = np.count_nonzero(diff, axis=0)
    reaction_time = y_output.shape[0] - diff_nonzero - start_t
    return reaction_time


# def write_test_log(trial_type, accuracies):
#     with open(join(par['save_dir'], par['log_fn']), 'a') as file:
#         file.write('\n--------------------------------\n')
#         file.write(par['synapse_config'] + ':\n')
#         file.write(trial_type + '\n')
#         file.write('H:' + str(np.mean(accuracies['H'])) + '\n')
#         file.write('M:' + str(np.mean(accuracies['M'])) + '\n')
#         file.write('L:' + str(np.mean(accuracies['L'])) + '\n')
#         file.write('N:' + str(np.mean(accuracies['N'])) + '\n')
