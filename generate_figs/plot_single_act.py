
import numpy as np 
import tables
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# plot settings
plt.rcParams['figure.figsize'] = [6, 4]
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['lines.linewidth'] = 2

f_dir = 'F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model'
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
    motion_rng = np.concatenate((np.arange(0, 40), np.arange(80, 120), np.arange(160, 170), np.arange(180, 190)), axis=0)
    target_rng = np.concatenate((np.arange(40, 80), np.arange(120, 160), np.arange(170, 180), np.arange(190, 200)), axis=0)
    for i in motion_rng:
        plot_avgAct_combined(h, y, desired_out, stim_level, stim_dir, i, True, mode='motion')
    for i in target_rng:
        plot_avgAct_combined(h, y, desired_out, stim_level, stim_dir, i, True, mode='target')

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


def plot_avgAct_combined(h, y, desired_out, stim_level, stim_dir, cell_idx, save_plt, mode):
    fig, ax = plt.subplots()
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    red_idx = stim_dir==315
    green_idx = stim_dir==135
    choice_color = get_choice_color(y, desired_out, stim_dir) # return choice color (green = 0, red = 1)
    correct_idx = find_correct_idx(y, desired_out)
    
    # plot lines
    colors = {'H_red':'#FF0000', 'M_red': '#B30000', 'L_red': '#660000', 'H_green': '#00FF00', 'M_green': '#00B300', 'L_green':'#006600'}
    # zero coherence stimulus direction is based on the choice color
    if mode == "motion":
        if sum(Z_idx) != 0:
            ax.plot(np.mean(h[19:, combine_idx(Z_idx, choice_color[-1, :]==0), cell_idx], axis=1), linestyle ='--', color='#000000', label='135, Z')
            ax.plot(np.mean(h[19:, combine_idx(Z_idx, choice_color[-1, :]==1), cell_idx], axis=1), color='#000000', label='315, Z')
        ax.plot(np.mean(h[19:, combine_idx(M_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['M_green'], label='135, M')
        ax.plot(np.mean(h[19:, combine_idx(L_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['L_green'], label='135, L')
        ax.plot(np.mean(h[19:, combine_idx(H_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['H_green'], label='135, H')
        
        ax.plot(np.mean(h[19:, combine_idx(L_idx, correct_idx, red_idx), cell_idx], axis=1), color=colors['L_red'], label='315, L')
        ax.plot(np.mean(h[19:, combine_idx(M_idx, correct_idx, red_idx), cell_idx], axis=1), color=colors['M_red'], label='315, M')
        ax.plot(np.mean(h[19:, combine_idx(H_idx, correct_idx, red_idx), cell_idx], axis=1), color=colors['H_red'], label='315, H')
    elif mode == 'target':
        red_choice = choice_color[-1, :]==1
        green_choice =  choice_color[-1, :]==0
        if sum(Z_idx) != 0:
            ax.plot(np.mean(h[19:, combine_idx(Z_idx, green_choice), cell_idx], axis=1), linestyle ='--', color='#000000', label='green targ, Z')
            ax.plot(np.mean(h[19:, combine_idx(Z_idx, red_choice), cell_idx], axis=1), color='#000000', label='red targ, Z')
        ax.plot(np.mean(h[19:, combine_idx(L_idx, correct_idx, green_choice), cell_idx], axis=1), color=colors['L_green'], label='green targ, L')
        ax.plot(np.mean(h[19:, combine_idx(M_idx, correct_idx, green_choice), cell_idx], axis=1), color=colors['M_green'], label='green targ, M')
        ax.plot(np.mean(h[19:, combine_idx(H_idx, correct_idx, green_choice), cell_idx], axis=1), color=colors['H_green'], label='green targ, H')
        
        ax.plot(np.mean(h[19:, combine_idx(L_idx, correct_idx, red_choice), cell_idx], axis=1), color=colors['L_red'], label='red targ, L')
        ax.plot(np.mean(h[19:, combine_idx(M_idx, correct_idx, red_choice), cell_idx], axis=1), color=colors['M_red'], label='red targ, M')
        ax.plot(np.mean(h[19:, combine_idx(H_idx, correct_idx, red_choice), cell_idx], axis=1), color=colors['H_red'], label='red targ, H')
    
    
    ax.set_xlim(0, 50)
    xticks = np.array([0, 25, 50])
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks+20-stim_st_time)*20)

    ax.set_title('Cell %d, %s'% (cell_idx, mode))
    ax.set_ylabel("Average activity")
    ax.set_xlabel("Time")
    ax.axvline(x=target_st_time-20, color='k', alpha=0.8, linestyle='--', linewidth=1)
    ax.axvline(x=stim_st_time-20, color='k', alpha=0.8, linestyle='--', linewidth=1)
    ax.legend(loc='best', prop={'size': 10}, frameon=False)
    plt.tight_layout()

    if save_plt:
        pic_dir = os.path.join(f_dir, 'single_neuron_activity_rep%d_combined' %(rep))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir,'cell_%d.png'% cell_idx))
        plt.savefig(os.path.join(pic_dir,'cell_%d.pdf'% cell_idx))
        plt.close(fig)



def plot_coh_avgAct(h, y, desired_out, stim_level, stim_dir, cell_idx, save_plt, mode):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize =(10, 4))
    H_idx, M_idx, L_idx, Z_idx = find_coh_idx(stim_level)
    red_idx = stim_dir==315
    green_idx = stim_dir==135
    choice_color = get_choice_color(y, desired_out, stim_dir)
    contra_idx, ipsi_idx = find_sac_idx(y)
    correct_idx = find_correct_idx(y, desired_out)
    
    # plot lines


    colors = {'H_red':'#FF0000', 'M_red': '#B30000', 'L_red': '#660000', 'H_green': '#00FF00', 'M_green': '#00B300', 'L_green':'#006600'}
    # zero coherence stimulus direction is based on the choice color
    if sum(Z_idx) != 0:
        ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], Z_idx, choice_color[-1, :]==0), cell_idx], axis=1), linestyle ='--', color='#000000', label='135, Z')
        ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], Z_idx, choice_color[-1, :]==1), cell_idx], axis=1), color='#000000', label='315, Z')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], L_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['H_green'], label='135, L')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], M_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['M_green'], label='135, M')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], H_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['L_green'], label='135, H')
    
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], L_idx, correct_idx, red_idx), cell_idx], axis=1), color=colors['H_red'], label='315, L')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], M_idx, correct_idx, red_idx), cell_idx], axis=1), color=colors['M_red'], label='315, M')
    ax1.plot(np.mean(h[:, combine_idx(ipsi_idx[-1, :], H_idx, correct_idx, red_idx), cell_idx], axis=1), color=colors['L_red'], label='315, H')
    ax1.set_title("Ipsi-lateral Saccade")
    ax1.set_ylabel("Average activity")
    ax1.set_xlabel("Time")
    ax1.axvline(x=target_st_time, color='k')
    ax1.axvline(x=stim_st_time, color='k')

    if sum(Z_idx) != 0:
        ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], Z_idx, choice_color[-1, :]==0), cell_idx], axis=1), linestyle ='--', color='#000000', label='135, Z')
        ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], Z_idx, choice_color[-1, :]==1), cell_idx], axis=1), color='#000000', label='315, Z')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], L_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['H_green'], label='135, L')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], M_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['M_green'], label='135, M')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], H_idx, correct_idx, green_idx), cell_idx], axis=1), color=colors['L_green'], label='135, H')
    
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], L_idx, correct_idx, red_idx), cell_idx], axis=1), color = colors['H_red'], label='315, L')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], M_idx, correct_idx, red_idx), cell_idx], axis=1), color = colors['M_red'], label='315, M')
    ax2.plot(np.mean(h[:, combine_idx(contra_idx[-1, :], H_idx, correct_idx, red_idx), cell_idx], axis=1), color = colors['L_red'], label='315, H')
    ax2.set_title("Contra-lateral Saccade")
    ax2.set_xlabel("Time")
    ax2.legend(loc='best', prop={'size': 10})
    ax2.axvline(x=target_st_time, color='k')
    ax2.axvline(x=stim_st_time, color='k')

    plt.suptitle('Cell %d, %s'% (cell_idx, ))

    if save_plt:
        pic_dir = os.path.join(f_dir, 'single_neuron_activity_rep%d_lr%f' %(rep, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir,'cell_%d.png'% cell_idx, mode))
       
        plt.close(fig)


main()