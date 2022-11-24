import numpy as np 
import tables
import matplotlib.pyplot as plt
import os
from pickle import dump, load

f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_shufFeedback_model"
model_type = f_dir.split('_')[-2]
total_rep = 3
total_shuf = 3
lr = 0.02

stim_st_time = 45
dt = 20

shuf_files = True
re_load_files = True

threshold = 3

def find_rt(diff, N=3):
    diff[:stim_st_time] = 0
    mask = np.convolve(diff,np.ones(N,dtype=int))>=N
    if mask.any():
        return (mask.argmax() - N + 1 - stim_st_time)*dt
    else:
        return (len(diff) - stim_st_time)*20

def load_files():
    if not shuf_files:
        for rep in range(total_rep):
            test_output = tables.open_file(os.path.join(f_dir, 'test_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
            test_table = test_output.root
            y_hist = test_table['y_hist_iter0'][:]
            if rep==0:
                all_y_hist = y_hist
            else:
                all_y_hist = np.concatenate((all_y_hist, y_hist), axis=1)
    else:
        for rep in range(total_rep):
            for shuf in range(total_shuf):
                test_output = tables.open_file(os.path.join(f_dir, 'test_output_lr%f_rep%d_shuf%d.h5'%(lr, rep, shuf)), mode = 'r')
                test_table = test_output.root
                y_hist = test_table['y_hist_iter0'][:]
                if rep==0 and shuf==0:
                    all_y_hist = y_hist
                else:
                    all_y_hist = np.concatenate((all_y_hist, y_hist), axis=1)
    dump(all_y_hist, open(os.path.join(f_dir, '%s_all_y_hist.pkl'%model_type), 'wb'))
    return all_y_hist
        

def main():
    if re_load_files:
        all_y_hist = load_files()
    else:
        all_y_hist = load(open(os.path.join(f_dir, '%s_all_y_hist.pkl'%model_type), 'rb'))
    act_diff = abs(all_y_hist[:, :, 0] - all_y_hist[:, :, 1])
    diff = act_diff > threshold
    all_rt = np.apply_along_axis(find_rt, 0, diff)
    dump(all_rt, open(os.path.join(f_dir, '%s_all_rt.pkl'%model_type), 'wb'))
    plt.hist(all_rt, bins = 20)
    plt.show()

if __name__ == '__main__':
    main()