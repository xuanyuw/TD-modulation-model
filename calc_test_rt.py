import numpy as np 
import tables
import matplotlib.pyplot as plt
import os
import pickle

f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
rep = 0
lr = 0.02

stim_st_time = 45
target_st_time = 25

test_output = tables.open_file(os.path.join(f_dir, 'test_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
test_table = test_output.root
y_hist = test_table['y_hist_iter0'][:]