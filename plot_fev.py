import numpy as np
from plot_activities import find_sac_idx, find_cell_group, min_max_normalize, get_max_iter
import os
import matplotlib.pyplot as plt
import tables

f_dir = 'new_input_model'
rep = 0
lr = 0.02

stim_st_time = 45
target_st_time = 25


def main():
    test_output = tables.open_file(os.path.join(f_dir, 'test_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
    # train_table = train_output.root
    test_table = test_output.root
    max_iter = get_max_iter(test_table)
    h = test_table['h_iter%d' %max_iter][:]
    normalized_h = min_max_normalize(h)
    y = test_table['y_hist_iter%d' %max_iter][:]
    stim_dir = test_table['stim_dir_iter%d' %max_iter][:]
    # plot single neural fev
    plot_cell_fev(normalized_h, y, stim_dir, True)

def calculate_fev(h0, left_ind, right_ind):

    h0_cut = np.dstack([h0[i:1+i-5 or None:1] for i in range(5)])
    gm = np.mean(h0_cut, axis=(1, 2))
    # left_mean = np.mean(h0_cut[:,left_ind,:], axis=(1, 2))
    # right_mean = np.mean(h0_cut[:,right_ind,:], axis=(1, 2))

    left_mean = np.mean(h0_cut[:,left_ind,:], axis=(1, 2))
    right_mean = np.mean(h0_cut[:,right_ind,:], axis=(1, 2))

    ssq_total = np.sum(np.square(h0_cut - gm[:, None, None]), axis=(1, 2))
    ssq_cond = sum(left_ind) * np.square(left_mean - gm) + \
                sum(right_ind) * np.square(right_mean - gm)
    
    mse = (ssq_total - ssq_cond) / (h0_cut.shape[1] - 2)
    fev = (ssq_cond-mse)/(ssq_total+mse)
    return fev

def plot_cell_fev(h, y, stim_dir, save_plt):
    contra_idx, ipsi_idx = find_sac_idx(y)
    temp_h = np.transpose(h, (2, 0, 1))
    all_fev_saccade = list(map(lambda h0: calculate_fev(h0, contra_idx[-1, :], ipsi_idx[-1, :]), temp_h))
    all_fev_motion = list(map(lambda h0: calculate_fev(h0, stim_dir==315, stim_dir==135), temp_h))

    for i in range(h.shape[2]):
        plt.plot(all_fev_saccade[i], label='saccade')
        plt.plot(all_fev_motion[i], label='motion')
        plt.title('FEV of cell %d, %s'%(i, find_cell_group(i)))
        plt.legend()
        plt.axvline(target_st_time - 2.5, color='black')
        plt.axvline(stim_st_time - 2.5, color='black')
        if save_plt:
            pic_dir = os.path.join(f_dir, 'single_neuron_fev')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            plt.savefig(os.path.join(pic_dir,'cell_%d.png'% i))
            plt.close()

main()