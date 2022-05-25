
from numpy import load
from os.path import join
import brainpy as bp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import TwoSlopeNorm

w = load('largeIter_model/weight_2_lr0.001000.pth', allow_pickle=True)
w = w.item()
w_in = w['w_in0']
w_out = w['w_out0']
w_rnn = w['w_rnn0']
in_mask = w['in_mask_init']
out_mask = w['out_mask_init']
rnn_mask = w['rnn_mask_init']


def plot_weights(weights, title, show_rnn_weights=False):
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
    plt.savefig(title+'.pdf', format='pdf')
    plt.show()


# plot_weights(w_in*in_mask, 'Input Weight', show_rnn_weights=False)
# plot_weights(w_out*out_mask, 'Output Weight', show_rnn_weights=False)
plot_weights(w_rnn*rnn_mask, 'RNN Weight', show_rnn_weights=True)
