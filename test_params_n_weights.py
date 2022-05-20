import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from init_weight import all_weights


def plot_weights(weights, title, show_rnn_weights=False):
    if show_rnn_weights:
        f = plt.figure(figsize=(8, 6))
    else:
        f = plt.figure(figsize=(8, 4))
    ax = f.add_subplot(1, 1, 1)
    im = ax.imshow(weights, aspect='auto', interpolation='none')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    f.colorbar(im, orientation='vertical')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('From')
    ax.set_xlabel('T0')
    ax.set_title(title)
    plt.savefig(title+'.pdf', format='pdf')
    plt.show()


plot_weights(all_weights['in_mask_init'], 'Input_Mask')
plot_weights(all_weights['out_mask_init'], 'Output_Mask')
plot_weights(all_weights['rnn_mask_init'], 'RNN_Mask', show_rnn_weights=True)
