"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
import torch
from os.path import join


def get_perf(target, output, mask):
    """ Calculate task accuracy by comparing the actual network output to the desired output
        only examine time points when test stimulus is on, e.g. when y[:,:,0] = 0 """

    mask_full = np.float32(mask > 0)
    target_max = np.argmax(target, axis=2)
    output_max = torch.argmax(output, axis=2).cpu().detach().numpy()
    stim_level = par['stim_level']
    accuracy = {'H': [], 'M': [], 'L': [], 'N': []}
    #print(target.shape, output.shape, mask.shape)
    #print(target[:,2,:], output[:,2,:])
    for i in range(target_max.shape[0]):
        trial_accuracy = np.sum(np.float32(
            target_max[i, :] == output_max[i, :])*mask_full[i, :])/np.sum(mask_full[i, :])
        if stim_level[i] == 'H':
            accuracy['H'].append(trial_accuracy)
        elif stim_level[i] == 'M':
            accuracy['M'].append(trial_accuracy)
        elif stim_level[i] == 'L':
            accuracy['L'].append(trial_accuracy)
        else:
            accuracy['N'].append(trial_accuracy)
    accuracy['H'] = np.array(accuracy['H'])
    accuracy['M'] = np.array(accuracy['M'])
    accuracy['L'] = np.array(accuracy['L'])
    accuracy['N'] = np.array(accuracy['N'])
    total_accuracy = np.sum(np.float32(
        target_max == output_max)*mask_full)/np.sum(mask_full)

    return accuracy, total_accuracy


def get_reaction_time(y_output):
    relu = torch.nn.ReLU(inplace=True)
    start_t = (par['time_fixation'] + par['time_target'])//par['dt']
    #reaciton_time = np.zeros((par['batch_size'], ))
    rt_mask = np.zeros(
        ((par['time_fixation'] + par['time_target'] + par['time_stim'])//par['dt'], 1))
    rt_mask[:, -par['time_stim']//par['dt']:] = 1
    rt_mask = torch.tensor(np.repeat(rt_mask, par['batch_size'], axis=1))
    abs_diff = torch.abs(y_output[:, :, 0] - y_output[:, :, 1]).cpu().detach()
    diff = torch.mul(rt_mask.T, relu(abs_diff - par['decision_threshold']))
    diff_nonzero = np.count_nonzero(diff.detach().numpy, axis=0)
    reaction_time = y_output.shape[0] - diff_nonzero - start_t
    return reaction_time


def write_test_log(trial_type, accuracies):
    with open(join(par['save_dir'], par['log_fn']), 'a') as file:
        file.write('\n--------------------------------\n')
        file.write(par['synapse_config'] + ':\n')
        file.write(trial_type + '\n')
        file.write('H:' + str(np.mean(accuracies['H'])) + '\n')
        file.write('M:' + str(np.mean(accuracies['M'])) + '\n')
        file.write('L:' + str(np.mean(accuracies['L'])) + '\n')
        file.write('N:' + str(np.mean(accuracies['N'])) + '\n')
