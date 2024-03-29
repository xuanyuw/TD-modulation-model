from tables import open_file
from os.path import join
from scipy.special import softmax
import numpy as np

def calc_coh_rt(root_dir, rep, shuf=None, decision_thresh=0.8):
    if shuf is None:
        test_output = open_file(join(root_dir, "test_output_lr0.020000_rep%d.h5"%rep), mode='r')
    else:
        test_output = open_file(join(root_dir, "test_output_lr0.020000_rep%d_shuf%d.h5"%(rep, shuf)), mode='r')
    
    test_output = test_output.root
    stim_level = np.array([i.decode("utf-8") for i in test_output['stim_level_iter0'][:]])
    y_hist = test_output['y_hist_iter0'][:]
    y = softmax(y_hist, axis=-1)
    y_diff = np.subtract(y[:, :, 0], y[:, :, 1])
    y_diff[:45, :] = 0
    rt = np.array(list(map(find_rt, (y_diff>decision_thresh).T)))
    return {'H': rt[stim_level=='H'], 'M': rt[stim_level=='M'], 'L': rt[stim_level=='L'], 'Z': rt[stim_level=='Z']}

    
def find_rt(arr, N=3, stim_onset = 45, dt = 20):
    temp_li = np.array([i for i in range(len(arr)) if all(arr[i:i+N]==True)])
    temp_li = temp_li[temp_li>stim_onset]
    if len(temp_li)==0: # set longer rt for trials that didn't reach threshold in the end
        return (len(arr) + 5 - stim_onset)*dt
    else:
        return (temp_li[0] - stim_onset)*dt
