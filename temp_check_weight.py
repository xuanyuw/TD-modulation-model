import numpy as np
import matplotlib.pyplot as plt
import os

f_dir = "2xMotionTuning_2xMotionNoise_weightCost_model"
all_rep = range(3)
all_lr = [0.02]

pic_dir = os.path.join(f_dir, "interneuron_weights")

def main(rep, lr):
    with open(os.path.join(f_dir, "weight_%d_lr%f.pth" % (rep, lr)), "rb") as f:
        all_weights = np.load(f, allow_pickle=True)
    all_weights = all_weights.item()
    w_rnn = all_weights['w_rnn0']

    with open(os.path.join(f_dir, "init_weight_%d_lr%f.pth" % (rep, lr)), "rb") as f:
        init_weights = np.load(f, allow_pickle=True)
    init_weights = init_weights.item()
    w_rnn_init = init_weights['w_rnn0']
    w_inter_hist(w_rnn, 'Trained_interneuron_weights_rep%d'%rep, pic_dir, True,)
    w_inter_hist(w_rnn-w_rnn_init, 'Diff_interneuron_weights_rep%d'%rep, pic_dir, True,)

    
def w_inter_hist(w_rnn, title, pic_dir, save_plt = False):
    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    d1 = w_rnn[:160, 200:].flatten()
    d1 = d1[d1!=0]
    ax1.hist(d1, bins = 25)
    ax1.set_title('Incoming weights')
    d2 = w_rnn[200:, :200].flatten()
    d2 = d2[d2!=0]
    ax2.hist(d2, bins = 25)
    ax2.set_title('Outgoing weights')
    plt.suptitle(title)
    if save_plt:
        plt.savefig(os.path.join(pic_dir, "%s.png" % title))
        plt.close(fig)
    else:
        plt.show()

    
for rep in all_rep:
    for lr in all_lr:
        main(rep, lr)