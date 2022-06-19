
import numpy as np 
import tables
import matplotlib.pyplot as plt
import os


f_dir = 'new_input_model'
rep = 0
lr = 0.02

stim_st_time = 45
target_st_time = 25

def main():

    # train_output = tables.open_file(os.path.join(f_dir, 'train_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
    test_output = tables.open_file(os.path.join(f_dir, 'test_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
    # train_table = train_output.root
    test_table = test_output.root
    max_iter = get_max_iter(test_table)
    h = test_table['h_iter%d' %max_iter][:]
    y = test_table['y_hist_iter%d' %max_iter][:]
    desired_out = test_table['target_iter%d' %max_iter][:]
    stim_level = test_table['stim_level_iter%d' %max_iter][:]
    stim_dir = test_table['stim_dir_iter%d' %max_iter][:]
    # plot population neural activity
    normalized_h = min_max_normalize(h)
    temp_h = normalized_h[:, :, np.append(range(0, 32), range(80, 112))]
    plot_population_activity(temp_h, y, desired_out, stim_level, stim_dir, 'Motion_Excitatory_Population', True)
    temp_h = normalized_h[:, :, np.append(range(32, 80), range(112, 160))]
    plot_population_activity(temp_h, y, desired_out, stim_level, stim_dir, 'Target_Excitatory_Population', True)

    temp_h = normalized_h[:, :, np.append(range(160, 168), range(180, 188))]
    plot_population_activity(temp_h, y, desired_out, stim_level, stim_dir, 'Motion_Inhibitory_Population', True)
    temp_h = normalized_h[:, :, np.append(range(168,180), range(188, 200))]
    plot_population_activity(temp_h, y, desired_out, stim_level, stim_dir, 'Target_Inhibitory_Population', True)

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
    module_bnd = y.shape[2]//2
    contra_idx[module_bnd:] = choice[module_bnd:]==1
    ipsi_idx[module_bnd:] = choice[module_bnd:]==0    
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

###################################
# Plot population neuron activity #
###################################

def find_pref_dir(stim_dir, h):
    red_idx = stim_dir==315
    green_idx = stim_dir==135
    red_mean = np.mean(h[stim_st_time:, red_idx, :], axis=(0, 1))
    green_mean = np.mean(h[stim_st_time:, green_idx, :], axis=(0, 1))
    pref_red = red_mean>green_mean
    return pref_red
    
def plot_population_activity(h, y, desired_out, stim_level, stim_dir, title, save_plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(10, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    # find the trial of preferred direction
    pref_red = find_pref_dir(stim_dir, h)
    dir_red = stim_dir==315
    pref_red_temp = np.tile(pref_red, (len(dir_red), 1))
    dir_red_temp = np.tile(np.reshape(dir_red, (-1, 1)), (1, len(pref_red)))
    pref_dir = pref_red_temp == dir_red_temp  

    contra_idx, ipsi_idx = find_sac_idx(y)
    correct_idx = find_correct_idx(y, desired_out)

    # zero coherence stimulus direction is based on the choice color
    if sum(Z_idx) != 0:
        ax1.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(ipsi_idx[-1, :], Z_idx)[:, None], pref_dir.shape))==0], axis=1), linestyle ='--', color='k', label='nonPref, Z')
        ax1.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(ipsi_idx[-1, :], Z_idx)[:, None], pref_dir.shape))==1], axis=1), color='k', label='Pref, Z')
    ax1.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(ipsi_idx[-1, :], L_idx, correct_idx)[:, None], pref_dir.shape))==0], axis=1), linestyle ='--', color='b', label='nonPref, L')
    ax1.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(ipsi_idx[-1, :], M_idx, correct_idx)[:, None], pref_dir.shape))==0], axis=1), linestyle ='--', color='g', label='nonPref, M')
    ax1.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(ipsi_idx[-1, :], H_idx, correct_idx)[:, None], pref_dir.shape))==0], axis=1), linestyle ='--', color='r', label='nonPref, H')
    
    ax1.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(ipsi_idx[-1, :], L_idx, correct_idx)[:, None], pref_dir.shape))==1], axis=1), color='b', label='Pref, L')
    ax1.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(ipsi_idx[-1, :], M_idx, correct_idx)[:, None], pref_dir.shape))==1], axis=1), color='g', label='Pref, M')
    ax1.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(ipsi_idx[-1, :], H_idx, correct_idx)[:, None], pref_dir.shape))==1], axis=1), color='r', label='Pref, H')
    ax1.set_title("Ipsi-lateral Saccade")
    ax1.set_ylabel("Average activity")
    ax1.set_xlabel("Time")
    ax1.axvline(x=target_st_time, color='k')
    ax1.axvline(x=stim_st_time, color='k')

    if sum(Z_idx) != 0:
        ax2.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(contra_idx[-1, :], Z_idx)[:, None], pref_dir.shape))==0], axis=1), linestyle ='--', color='k', label='nonPref, Z')
        ax2.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(contra_idx[-1, :], Z_idx)[:, None], pref_dir.shape))==1], axis=1), color='k', label='Pref, Z')
    ax2.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(contra_idx[-1, :], L_idx, correct_idx)[:, None], pref_dir.shape))==0], axis=1), linestyle ='--', color='b', label='nonPref, L')
    ax2.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(contra_idx[-1, :], M_idx, correct_idx)[:, None], pref_dir.shape))==0], axis=1), linestyle ='--', color='g', label='nonPref, M')
    ax2.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(contra_idx[-1, :], H_idx, correct_idx)[:, None], pref_dir.shape))==0], axis=1), linestyle ='--', color='r', label='nonPref, H')
    
    ax2.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(contra_idx[-1, :], L_idx, correct_idx)[:, None], pref_dir.shape))==1], axis=1), color='b', label='Pref, L')
    ax2.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(contra_idx[-1, :], M_idx, correct_idx)[:, None], pref_dir.shape))==1], axis=1), color='g', label='Pref, M')
    ax2.plot(np.mean(h[:,(pref_dir + np.broadcast_to(combine_idx(contra_idx[-1, :], H_idx, correct_idx)[:, None], pref_dir.shape))==1], axis=1), color='r', label='Pref, H')
    ax2.set_title("Contra-lateral Saccade")
    ax2.set_ylabel("Average activity")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.axvline(x=target_st_time, color='k')
    ax2.axvline(x=stim_st_time, color='k')

    plt.suptitle(title)

    if save_plt:
        pic_dir = os.path.join(f_dir, 'new_population_neuron_activity_rep%d_lr%f' %(rep, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir,'%s.png'% title))
        plt.close(fig) 

def min_max_normalize(arr):
    norm_arr = (arr - np.min(np.mean(arr,axis=1), axis=0))/(np.max(np.mean(arr,axis=1), axis=0) - np.min(np.mean(arr,axis=1), axis=0))
    return norm_arr


main()