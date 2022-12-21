from tables import open_file
from os.path import join
from scipy.special import softmax
import numpy as np

def calc_coh_rt(root_dir, rep, decision_thresh):
    test_output = open_file(join(root_dir, "test_output_lr0.020000_rep%d.h5"%rep), mode='r')
    test_output = test_output.root
    stim_level = np.array([i.decode("utf-8") for i in test_output['stim_level_iter0'][:]])
    y_hist = test_output['y_hist_iter0'][:]
    y = softmax(y_hist, axis=-1)
    y_diff = np.subtract(y[:, :, 0], y[:, :, 1])
    rt = np.array(list(map(find_rt, (y_diff>decision_thresh).T)))
    return {'H': rt[stim_level=='H'], 'M': rt[stim_level=='M'], 'L': rt[stim_level=='L'], 'Z': rt[stim_level=='Z']}

    
def find_rt(arr, N=3, stim_onset = 45, dt = 20):
    temp_li = np.array([i for i in range(len(arr)) if all(arr[i:i+N]==True)])
    temp_li = temp_li[temp_li>stim_onset]
    if len(temp_li)==0:
        return (len(arr) - stim_onset)*20
    else:
        return (temp_li[0] - stim_onset)*20
