from numpy.random import f
import torch
import time
import pickle
import os
import numpy as np
from parameters import *
from stimulus import *
from analysis import *
import model as _model
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt


##################
# Train & Test
##################

def train(model, opt, trial_info, device):
    """
    main train step
    """
    opt = torch.optim.AdamW(model.parameters(), lr=par['learning_rate'])
    neural_input = torch.tensor(trial_info['neural_input']).to(device)
    desired_output = torch.tensor(trial_info['desired_output']).to(device)
    train_mask = torch.tensor(trial_info['train_mask']).to(device)
    y_output, x = model(neural_input, desired_output, train_mask)
    model_loss = optimize(model, opt, device)

    # clear grad and temp var
    del neural_input, desired_output, train_mask

    return y_output, x, model_loss, trial_info


def test(model, trial_info, device):
    """
    main train step
    """

    model.eval()
    with torch.no_grad():
        neural_input = torch.tensor(trial_info['neural_input']).to(device)
        desired_output = torch.tensor(trial_info['desired_output']).to(device)
        train_mask = torch.tensor(trial_info['train_mask']).to(device)
        y_output, x = model(
            neural_input, desired_output, train_mask, train=False)
    return y_output, x


def train_loop(device, save_out=par['save_train_out']):
    """
    train a single model for number of times specified in par and save model info data if needed
    """
    # initialize weights
    initialize_weights()

    # initialize model
    model = _model.Model(device)

    # model.to(device)
    t0 = time.time()
    if save_out:
        # trial_info_list = []
        rt_list = []

    model_performance = {'accuracies': [],
                         'total_accuracy': [],  'loss': [], 'iteration': []}

    if par['mixed_train']:
        trial_types = np.random.choice(par['task_list'], par['num_iterations'])
    else:
        trial_types = [par['trial_type']] * par['num_iterations']
    model_performance['trial_types'] = trial_types

    model.load_weights()
    # initialize the fixed weights
    # w_rnn_base = torch.multiply(model.w_rnn_base, torch.tensor(par['conn_mask0']).to(device)).float()
    model.w_rnn_fix_t1r[par['t1r_target_in_idx'], :] = model.w_rnn_base[par['t1r_target_in_idx'], :]
    model.w_rnn_fix_t2r[par['t2r_target_in_idx'], :] = model.w_rnn_base[par['t2r_target_in_idx'], :]

    # opt = torch.optim.AdamW(model.parameters(), lr=par['learning_rate'])
    opt = None

    all_loss = []
    all_accs = []
    for i in range(par['num_iterations']):
        # pick a random task every iteration
        trial_type = trial_types[i]
        # generate stimulus for this iteration
        stim = Stimulus()
        update_parameters({'trial_type': trial_type,
                           'coherence_levels': par['train_coherence_levels'],
                           'pure_noise_perc': 0
                           })
        trial_info = stim.generate_trial()
        if stim.color_rf[0] == 1:  # t1 is red
            model.cond = 't1r'
        else:
            model.cond = 't2r'

        # train model
        y_output, x, model_loss, trial_info = train(
            model, opt, trial_info, device)

        if save_out:
            # trial_info_list.append(trial_info)
            rt_list.append(get_reaction_time(y_output))
        par['stim_level'] = trial_info['stim_level']
        agg_y_output = aggregate_out_activities(y_output).to(device)
        accuracies, total_accuracy = get_perf(
            trial_info['desired_output'], agg_y_output, trial_info['train_mask'])
        model_performance = append_model_performance(
            model_performance, accuracies, total_accuracy, model_loss, i)
        # # early exit if the performance is high enough
        if np.array(model_performance['total_accuracy'][-150:]).mean() >= 0.95:
            break
        # output model performance to screen
        if (i+1) % par['iters_between_outputs'] == 0:
            relu = torch.nn.ReLU(inplace=True)
            print_results(model_performance, i, relu(x), t0)
            # print_gpu_usage(device)

        # plot loss
        all_loss.append(model_loss.cpu().detach().numpy())
        all_accs.append(total_accuracy)
    plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(np.array(all_loss).flatten())
    plt.title('loss')
    plt.subplot(212)
    plt.plot(all_accs)
    plt.title('acc')
    plt.tight_layout()

    plt.savefig('model%d_lr%f_size%d_alternate_update.png' %
                (par['rep'], par['learning_rate'], np.sum(par['module_sizes'])))
    plt.close()

    # save trained model weights
    if not os.path.exists(par['save_dir']):
        os.makedirs(par['save_dir'])
    weight_fn = par['save_dir'] + par['weight_fn']
    merged_dict = model.state_dict()
    merged_dict['w_in'] = model.w_in
    torch.save(merged_dict, weight_fn)

    # Save model info if needed
    if save_out:
        data_dict = {'x': x.cpu().detach().numpy(),
                     'y': y_output.cpu().detach().numpy(),
                     'rt': rt_list,
                     'trial_info': trial_info}
        # Save model and results
        save_results(model_performance, data_dict)


def test_loop(device):
    """
    test the model on all tasks
    """
    trial_info_list = []
    rt_list = []
    x_list = []
    y_output_list = []
    model_performance = {'accuracies': [],
                         'total_accuracy': [], 'iteration': []}
    for i in range(len(par['task_list'])):  # iterate through the task lists
        for j in range(par['num_test_iterations']):
            trial_type = par['task_list'][i]
            if trial_type == 'MDC':
                update_parameters({
                    'trial_type': trial_type,
                    'pure_noise_perc': 0,
                    'batch_size': par['test_batch_size']
                })
            else:
                update_parameters({
                    'trial_type': trial_type,
                    'coherence_levels': par['test_coherence_levels'],
                    'pure_noise_perc': 0.2,
                    'batch_size': par['test_batch_size']
                })

            # generate stimulus
            stim = Stimulus()
            trial_info = stim.generate_trial()
            trial_info_list.append(trial_info)

            model = _model.Model(device=device, train=False)
            if stim.color_rf[0] == 1:  # t1 is red
                model.cond = 't1r'
            else:
                model.cond = 't2r'

            # test the model
            y_output, x = test(model, trial_info, device)
            # save the output
            rt_list.append(get_reaction_time(y_output))
            x_list.append(x.cpu().detach().numpy())
            y_output_list.append(y_output.cpu().detach().numpy())

            par['stim_level'] = trial_info['stim_level']
            agg_y_output = aggregate_out_activities(y_output).to(device)
            accuracies, total_accuracy = get_perf(
                trial_info['desired_output'], agg_y_output, trial_info['train_mask'])

            model_performance['total_accuracy'].append(total_accuracy)
            model_performance['accuracies'].append(accuracies)
            # write output to log
            if par['log_test']:
                write_test_log(trial_type, accuracies)

    # Save it out
    data_dict = {'x': x_list,
                 'y': y_output_list,
                 'rt': rt_list,
                 'trial_info': trial_info_list
                 }
    # Save model and results
    save_results(model_performance, data_dict, save_fn=par['test_save_fn'])

#####################
# Optimize
#####################


def reluDerivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def aggregate_out_activities(y):
    """Assuming every choice is controlled by same number of neurons"""
    div = np.linspace(
        0, par['n_output'], num=par['n_choices']+1, endpoint=True).astype(np.int)
    agg_y = torch.zeros(y.shape[0], y.shape[1], par['n_choices'])
    for i in range(par['n_choices']):
        agg_y[:, :, i] = torch.mean(y[:, :, div[i]:div[i+1]], dim=2)
    return agg_y


def calc_loss(model):
    # loss = torch.nn.MSELoss(reduction='none')
    loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    # apply mask to determine which part of output should participate in loss calculation
    new_mask = model.mask.unsqueeze(2).repeat(1, 1, 2)
    agg_y = aggregate_out_activities(model.y).to(model.device)
    perf_loss = torch.mul(loss(agg_y, model.target_data), new_mask).mean()

    n = 2 if par['spike_regularization'] == 'L2' else 1
    spike_loss = torch.mean(model.r**n)
    relu = torch.nn.ReLU()
    weight_loss = torch.mean(relu(model.w_rnn_base)**n)
    model_loss = perf_loss + par['spike_cost'] * \
        spike_loss + par['weight_cost']*weight_loss
    return model_loss


def nanmean(x, dim=None):
    # TODO: impute tensor with mean
    value = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    return torch.mean(value, dim=dim)


def optimize(model, opt, device):
    # Calculate the loss functions and optimize the weights
    model_loss = calc_loss(model)
    #opt = torch.optim.SGD(model.parameters(), lr=par['learning_rate'])
    opt.zero_grad()
    model_loss.backward()
    # clipping gradient norm to avoid gradient explosion
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), par['clip_max_grad_val'])

    opt.step()
    model.x_grad = []
    relu = torch.nn.ReLU()
    model.w_rnn_base = Parameter(relu(model.w_rnn_base))
    if par['noisy_weight_update']:
        noise = (torch.ones_like(model.w_rnn_base) * par['noisy_weight_coef'])
        model.w_rnn_base = Parameter(torch.where(
            model.w_rnn_base == 0, noise, model.w_rnn_base))
    model.w_out = Parameter(torch.mul(model.w_out, torch.tensor(
        par['w_out_mask'], requires_grad=False).to(device)))
    if model.cond == 't1r':
        model.w_rnn_fix_t1r[par['t1r_target_in_idx'], :] = model.w_rnn_base[par['t1r_target_in_idx'], :]
    else:
        model.w_rnn_fix_t2r[par['t2r_target_in_idx'], :] = model.w_rnn_base[par['t2r_target_in_idx'], :]
    return model_loss

#####################
# Print & Save
#####################


def save_results(model_performance, results, save_fn=None):

    results = {'parameters': par, 'results': results}
    for k, v in model_performance.items():
        results[k] = v
    if save_fn is None:
        fn = par['save_dir'] + par['save_fn']
    else:
        fn = par['save_dir'] + save_fn
    pickle.dump(results, open(fn, 'wb'))
    print('Model results saved in ', fn)


def append_model_performance(model_performance, accuracy, total_accuracy, loss, iteration):

    model_performance['total_accuracy'].append(total_accuracy)
    model_performance['accuracies'].append(accuracy)
    model_performance['loss'].append(loss.cpu().detach().numpy())
    model_performance['iteration'].append(iteration)

    return model_performance


def print_results(model_performance, iter_num, h, t0):
    t1 = time.time()
    loss_tp = np.mean(
        model_performance['loss'][-par['iters_between_outputs']:])
    acc_tp = np.mean(
        model_performance['total_accuracy'][-par['iters_between_outputs']:])
    print(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy ' + '{:0.4f}'.format(acc_tp) +
          ' | Loss {:0.8f}'.format(loss_tp) + ' | Mean activity {:0.4f}'.format(np.mean(h.cpu().detach().numpy())))
    print(f"Elapsed time: {str(t1 - t0)}")
    t0 = time.time()


def print_important_params():

    important_params = ['rep', 'num_iterations', 'learning_rate', 'n_hidden', 'noise_rnn_sd', 'noise_in_sd', 'spike_cost',
                        'spike_regularization', 'weight_cost', 'test_cost_multiplier', 'trial_type',
                        'synapse_config']
    for k in important_params:
        print(k, ': ', par[k])


def print_gpu_usage(device):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
