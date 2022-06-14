
from numpy import load
from os.path import join, exists
from os import makedirs
import brainpy as bp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import TwoSlopeNorm
import tables


fdir = 'correctedOut_rf_model'
rep = 0
lr = 0.02

def find_max_iter(table):
    all_iters = []
    for row in table:
        iter_num = int(row.name.split('iter')[1])
        if iter_num not in all_iters:
            all_iters.append(iter_num)
        else:
            break
    max_iter = max(all_iters)
    return max_iter


def plot_weights(weights, title, fdir, show_rnn_weights=False):
    if show_rnn_weights:
        f = plt.figure(figsize=(8, 6))
    else:
        f = plt.figure(figsize=(8, 4))
    ax = f.add_subplot(1, 1, 1)
    plt.set_cmap('bwr')
    norm = TwoSlopeNorm(vcenter=0)
    im = ax.imshow(weights, aspect='auto', interpolation='none', norm=norm)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    f.colorbar(im, orientation='vertical')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('From')
    ax.set_xlabel('T0')
    ax.set_title(title)
    plt.savefig(join(fdir, title+'.png'), format='png')
    # plt.show()

w = load(join(fdir, 'init_weight_%d_lr%f.pth'%(rep, lr)), allow_pickle=True)
w = w.item()
train_output = tables.open_file(join(fdir, 'train_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
table = train_output.root
max_iter = find_max_iter(table)
w_in_after = table['w_in_iter%d' %max_iter]
w_out_after = table['w_out_iter%d' %max_iter]
w_rnn_after = table['w_rnn_iter%d' %max_iter]
w_in_init = w['w_in0']
w_out_init = w['w_out0']
w_rnn_init = w['w_rnn0']
in_mask = w['in_mask_init']
out_mask = w['out_mask_init']
rnn_mask = w['rnn_mask_init']

pic_dir = join(fdir, 'weight_matrices_rep%d_lr%f'%(rep, lr))
if not exists(pic_dir):
    makedirs(pic_dir)


plot_weights(w_in_init*in_mask, 'Input_Weight_Init', pic_dir, show_rnn_weights=False)
plot_weights(w_out_init*out_mask, 'Output_Weight_Init', pic_dir, show_rnn_weights=False)
plot_weights(w_rnn_init*rnn_mask, 'RNN_Weight_Init', pic_dir, show_rnn_weights=True)

plot_weights(w_in_after*in_mask, 'Input_Weight_After', pic_dir, show_rnn_weights=False)
plot_weights(w_out_after*out_mask, 'Output_Weight_After', pic_dir, show_rnn_weights=False)
plot_weights(w_rnn_after*rnn_mask, 'RNN_Weight_After', pic_dir, show_rnn_weights=True)
