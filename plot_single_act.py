from locale import normalize
import numpy as np 
import tables
import matplotlib.pyplot as plt
import os
import pickle

f_dir = 'new_input_model'
rep = 0
lr = 0.02

stim_st_time = 45
target_st_time = 25

def main():
    test_output = tables.open_file(os.path.join(f_dir, 'test_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
    test_table = test_output.root
    max_iter = get_max_iter(test_table)
    h = test_table['h_iter%d' %max_iter][:]
    y = test_table['y_hist_iter%d' %max_iter][:]
    desired_out = test_table['target_iter%d' %max_iter][:]
    stim_level = test_table['stim_level_iter%d' %max_iter][:]
    stim_dir = test_table['stim_dir_iter%d' %max_iter][:]
    # plot single neural activity
    for i in range(200):
        plot_coh_avgAct(h, y, desired_out, stim_level, stim_dir, i, True)
def relu(input):
    return input * (input>0)

def get_max_iter(table):
    all_iters = []
    for row in table:
        iter_num = int(row.name.split('iter')[1])
        if iter_num not in all_iters:
            all_iters.append(iter_num)
        else:
            break
    max_iter = max(all_iters)
    return max_iter

def find_cell_group(cell_idx):
    grp = ''
    if cell_idx in range(160):
        grp += 'Exc'
        if cell_idx in range(80):
            grp += '_Contra'
            if cell_idx in range(32):
                grp += '_Motion'
            else:
                grp += '_Target'
        else:
            grp += '_Ipsi'
            if cell_idx in range(80, 112):
                grp += '_Motion'
            else:
                grp += '_Target'
       
    else:
        grp += 'Inh'
        if cell_idx in range(160, 180):
            grp += '_Contra'
            if cell_idx in range(160, 168):
                grp += '_Motion'
            else:
                grp += '_Target'
        else:
            grp += '_Ipsi'
            if cell_idx in range(180, 188):
                grp += '_Motion'
            else:
                grp += '_Target'
    return grp

###############################
# Plot single neuron activity #
###############################

def find_coh_idx(stim_level):
    H_idx = np.array(stim_level)==b'H'
    M_idx = np.array(stim_level)==b'M'
    L_idx = np.array(stim_level)==b'L'
    Z_idx = np.array(stim_level)==b'Z'
    return H_idx, M_idx, L_idx, Z_idx

def find_sac_idx(y):
    choice = np.argmax(y, 2)
    contra_idx = choice==0
    ipsi_idx = choice==1
    return contra_idx, ipsi_idx

def recover_targ_loc(desired_out, stim_dir):
    # return the target arrangement: green_contra = 0, red_contra = 1
    choice = np.argmax(desired_out, 2)
    target = choice==1 # contra = 0, ipsi = 1
    dir = stim_dir==315 # green = 0, red = 1
    return np.logical_xor(target, dir)

def get_choice_color(y, desired_out, stim_dir):
    # return choice color (green = 0, red = 1)
    choice = np.argmax(y, 2)
    targ_loc = recover_targ_loc(desired_out, stim_dir)
    return np.logical_xor(choice, targ_loc)

def find_pref_dir(stim_dir, h):
    red_idx = stim_dir==315
    green_idx = stim_dir==135
    red_mean = np.mean(h[stim_st_time:, red_idx, :], axis=[0, 1])
    green_mean = np.mean(h[stim_st_time:, green_idx, :], axis=[0, 1])
    pref_red = red_mean>green_mean
    return pref_red

def calc_avg_idx(h, trial_idx, cell_idx):
    return np.mean(h[:, trial_idx, cell_idx], axis=[1, 2])

def find_correct_idx(y, desired_output, Z_idx):
    target_max = np.argmax(desired_output, axis=2)[-1, :]
    output_max = np.argmax(y, axis=2)[-1, :]
    return (target_max == output_max + Z_idx)

def find_coh_idx(stim_level):
    H_idx = np.array(stim_level)==b'H'
    M_idx = np.array(stim_level)==b'M'
    L_idx = np.array(stim_level)==b'L'
    Z_idx = np.array(stim_level)==b'Z'
    return H_idx, M_idx, L_idx, Z_idx

def find_sac_idx(y):
    choice = np.argmax(y, 2)
    contra_idx = choice==0
    ipsi_idx = choice==1
    return contra_idx, ipsi_idx

def recover_targ_loc(desired_out, stim_dir):
    # return the target arrangement: green_contra = 0, red_contra = 1
    choice = np.argmax(desired_out, 2)
    target = choice==1 # contra = 0, ipsi = 1
    dir = stim_dir==315 # green = 0, red = 1
    return np.logical_xor(target, dir)

def get_choice_color(y, desired_out, stim_dir):
    # return choice color (green = 0, red = 1)
    choice = np.argmax(y, 2)
    targ_loc = recover_targ_loc(desired_out, stim_dir)
    return np.logical_xor(choice, targ_loc)

def find_pref_dir(stim_dir, h):
    red_idx = stim_dir==315
    green_idx = stim_dir==135
    red_mean = np.mean(h[stim_st_time:, red_idx, :], axis=[0, 1])
    green_mean = np.mean(h[stim_st_time:, green_idx, :], axis=[0, 1])
    pref_red = red_mean>green_mean
    return pref_red

def calc_avg_idx(h, trial_idx, cell_idx):
    return np.mean(h[:, trial_idx, cell_idx], axis=[1, 2])

def find_correct_idx(y, desired_output):
    target_max = np.argmax(desired_output, axis=2)[-1, :]
    output_max = np.argmax(y, axis=2)[-1, :]
    return (target_max == output_max)

def combine_idx(*args):
    temp = args[0]
    for i in range(1, len(args)):
        temp = np.logical_and(temp, args[i])
    return temp  

def plot_coh_avgAct(h, y, desired_out, stim_level, stim_dir, cell_idx, save_plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(10, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    red_idx = stim_dir==315
    green_idx = stim_dir==135
    choice_color = get_choice_color(y, desired_out, stim_dir)
    contra_idx, ipsi_idx = find_sac_idx(y)
    correct_idx = find_correct_idx(y, desired_out)

    # zero coherence stimulus direction is based on the choice color
    if sum(Z_idx) != 0:
        ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], Z_idx, choice_color[-1, :]==0), cell_idx], axis=1), linestyle ='--', color='#000000', label='135, Z')
        ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], Z_idx, choice_color[-1, :]==1), cell_idx], axis=1), color='#000000', label='315, Z')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], L_idx, correct_idx, green_idx), cell_idx], axis=1), color='#046C41', label='135, L')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], M_idx, correct_idx, green_idx), cell_idx], axis=1), color='#1AA33C', label='135, M')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], H_idx, correct_idx, green_idx), cell_idx], axis=1), color='#83D475', label='135, H')
    
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], L_idx, correct_idx, red_idx), cell_idx], axis=1), color='#910000', label='315, L')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], M_idx, correct_idx, red_idx), cell_idx], axis=1), color='#D32431', label='315, M')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], H_idx, correct_idx, red_idx), cell_idx], axis=1), color='#CE5757', label='315, H')
    ax1.set_title("Ipsi-lateral Saccade")
    ax1.set_ylabel("Average activity")
    ax1.set_xlabel("Time")
    ax1.legend()
    ax1.axvline(x=target_st_time, color='k')
    ax1.axvline(x=stim_st_time, color='k')

    if sum(Z_idx) != 0:
        ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], Z_idx, choice_color[-1, :]==0), cell_idx], axis=1), linestyle ='--', color='#000000', label='135, Z')
        ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], Z_idx, choice_color[-1, :]==1), cell_idx], axis=1), color='#000000', label='315, Z')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], L_idx, correct_idx, green_idx), cell_idx], axis=1), color='#046C41', label='135, L')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], M_idx, correct_idx, green_idx), cell_idx], axis=1), color='#1AA33C', label='135, M')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], H_idx, correct_idx, green_idx), cell_idx], axis=1), color='#83D475', label='135, H')
    
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], L_idx, correct_idx, red_idx), cell_idx], axis=1), color='#910000', label='315, L')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], M_idx, correct_idx, red_idx), cell_idx], axis=1), color='#D32431', label='315, M')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], H_idx, correct_idx, red_idx), cell_idx], axis=1), color='#CE5757', label='315, H')
    ax2.set_title("Contra-lateral Saccade")
    ax2.set_ylabel("Average activity")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.axvline(x=target_st_time, color='k')
    ax2.axvline(x=stim_st_time, color='k')

    plt.suptitle('Cell %d, %s'% (cell_idx, find_cell_group(cell_idx)))

    if save_plt:
        pic_dir = os.path.join(f_dir, 'single_neuron_activity_rep%d_lr%f' %(rep, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir,'cell_%d.png'% cell_idx))
        plt.close(fig)


main()