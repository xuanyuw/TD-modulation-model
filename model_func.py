from model import Model
from stimulus import Stimulus
from analysis import get_perf, get_reaction_time
from os.path import dirname, exists, join
from os import makedirs
from pickle import dump
from numpy import save, mean
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


bp.math.set_platform('cpu')


def trial(par, train=True, save_results=True,):
    num_iterations = par['num_iterations']
    iter_between_outputs = par['iters_between_outputs']
    stim = Stimulus(par)

    model = Model(par, train=train)
    opt = bp.optimizers.Adam(par['learning_rate'],
                             train_vars=model.train_vars())
    if train:
        grad_f = bm.grad(model.loss_func,
                         dyn_vars=model.vars(),
                         grad_vars=model.train_vars(),
                         return_value=True)

    @bm.jit
    @bp.math.function(nodes=(model, opt))
    def train_op(x, y, mask):
        grads, _ = grad_f(x, y, mask)
        capped_gs = dict()
        for key, grad in grads.items():
            if 'w_rnn' in key:
                grad *= model.rnn_mask
            elif 'w_out' in key:
                grad *= model.out_mask
            elif 'w_in' in key:
                grad *= model.in_mask
            capped_gs[key] = bm.clip_by_norm(grad, par['clip_max_grad_val'])
        opt.update(grads=capped_gs)

    # keep track of the model performance across training
    model_performance = {'total_accuracy': [], 'rt': [], 'loss': [], 'perf_loss': [],
                         'spike_loss': [], 'weight_loss': [], 'iteration': [], 'H_acc': [],
                         'M_acc': [], 'L_acc': [], 'Z_acc': []}

    for i in range(num_iterations):
        model.reset()
        # generate batch of batch_train_size
        stim = Stimulus(par)
        trial_info = stim.generate_trial()
        inputs = bm.array(trial_info['neural_input'], dtype=bm.float32)
        targets = bm.array(trial_info['desired_output'], dtype=bm.float32)
        mask = bm.array(trial_info['train_mask'], dtype=bm.float32)

        # Run the model
        if train:
            train_op(inputs, targets, mask)

        # get metrics
        accuracy, total_accuracy = get_perf(
            targets, model.y_hist, mask, trial_info['stim_level'])
        H_acc = accuracy['H']
        M_acc = accuracy['M']
        L_acc = accuracy['L']
        Z_acc = accuracy['Z']
        rt = mean(get_reaction_time(model.y_hist, par))
        model_performance['total_accuracy'].append(total_accuracy)
        model_performance['rt'].append(rt)
        model_performance['loss'].append(model.loss)
        model_performance['perf_loss'].append(model.perf_loss)
        model_performance['spike_loss'].append(model.spike_loss)
        model_performance['weight_loss'].append(model.weight_loss)
        model_performance['H_acc'].append(H_acc)
        model_performance['M_acc'].append(M_acc)
        model_performance['L_acc'].append(L_acc)
        model_performance['Z_acc'].append(Z_acc)

        # Save the network model and output model performance to screen
        if i % iter_between_outputs == 0:
            if train:
                print(f' Iter {i:4d}' +
                      f' | Accuracy {total_accuracy:0.4f}' +
                      f' | Perf loss {model.perf_loss[0]:0.4f}' +
                      f' | Spike loss {model.spike_loss[0]:0.4f}' +
                      f' | Weight loss {model.weight_loss[0]:0.4f}' +
                      f' | Mean activity {bm.mean(model.h):0.4f}' +
                      f' | RT {rt:0.4f}')
                print(f'Separated Accuracy:' +
                      f' | H {H_acc:0.4f}' +
                      f' | M {M_acc:0.4f}' +
                      f' | L {L_acc:0.4f}' +
                      f' | Z {Z_acc:0.4f}')
                print('--------------------------------------------------------------------------------------------------------------------------------')
            else:
                print(f' Iter {i:4d}' +
                      f' | Accuracy {total_accuracy:0.4f}' +
                      f' | Mean activity {bm.mean(model.h):0.4f}')
                print(f'Separated Accuracy:' +
                      f' | H {H_acc:0.4f}' +
                      f' | M {M_acc:0.4f}' +
                      f' | L {L_acc:0.4f}' +
                      f' | Z {Z_acc:0.4f}')
                print('--------------------------------------------------------------------------------------------------------------------------------')

    if save_results:
        if not exists(dirname(par['save_dir'])):
            makedirs(dirname(par['save_dir']))

    if train:
        plot_curves(model_performance['total_accuracy'], model_performance['H_acc'],
                    model_performance['M_acc'], model_performance['L_acc'], model_performance['Z_acc'], join(
                        par['save_dir'], 'Training_Accuracies.pdf'), par['learning_rate'], par['rep'])

        # Save model and results
        weights = {}
        w = model.train_vars().unique().dict()
        for k, v in w.items():
            temp = k.split('.')
            weights[temp[1] + '0'] = v

        # Save weight masks
        all_masks = model.get_all_masks()
        for k in all_masks.keys():
            weights[k] = all_masks[k]

        with open(join(par['save_dir'], par['weight_fn']), 'wb') as f:
            save(f, weights)
    results = {}
    for k, v in model_performance.items():
        results[k] = v
    dump(results, open(join(par['save_dir'], par['save_fn']), 'wb'))


def plot_curves(all_arr, h_arr, m_arr, l_arr, z_arr, fn, lr, rep):
    f = plt.figure(figsize=(8, 3))
    ax = f.add_subplot(1, 1, 1)

    ax.plot(h_arr, '#7CB9E8', alpha=0.6, label='high coh')
    ax.plot(m_arr, '#007FFF', alpha=0.6, label='mid coh')
    ax.plot(l_arr, '#00308F', alpha=0.6, label='low coh')
    ax.plot(z_arr, 'gray', alpha=0.6, label='zero coh')
    ax.plot(all_arr, 'r', linewidth=2.5, label='total')
    ax.legend()
    ax.set_title('Learning Rate = %f, Rep %d' % (lr, rep))
    plt.savefig(fn, format='pdf')
    plt.show()
