import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
from pickle import load

f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_shufFeedback_model"
total_rep = 50
total_shuf = 100
all_lr = [2e-2]
plot_shuf = True

def shuf_main():
    all_acc_df = pd.DataFrame(columns=['coh', 'acc'])
    for rep in range(total_rep):
        print('Plotting rep %d' % rep)
        rep_acc_df = pd.DataFrame(columns=['coh', 'acc'])
        for shuf in range(total_shuf):
            with open(join(f_dir, 'test_results_%d_shuf%d.pkl' %(rep, shuf)), 'rb') as f:
                data = load(f)
            acc_df = pd.DataFrame({
                "coh":['H', 'M', 'L', 'Z'], 
                "acc":[data['H_acc'][0], data['M_acc'][0], data['L_acc'][0], data['Z_acc'][0]]
                                    })
            all_acc_df = pd.concat([all_acc_df, acc_df], ignore_index=True)
            rep_acc_df = pd.concat([rep_acc_df, acc_df], ignore_index=True)
        plot_test_acc(rep_acc_df, f_dir, 0.02, rep)
    plot_test_acc(all_acc_df, f_dir, 0.02)
    acc_mean = all_acc_df.groupby('coh').mean()
    acc_std = all_acc_df.groupby('coh').std()
    acc_mean.to_csv(join(f_dir, 'test_acc_mean.csv'))
    acc_std.to_csv(join(f_dir, 'test_acc_std.csv'))
            

def no_shuf_main():
    all_acc_df = pd.DataFrame(columns=['coh', 'acc'])
    for rep in range(total_rep):
        with open(join(f_dir, 'test_results_%d.pkl' %(rep)), 'rb') as f:
            data = load(f)
        acc_df = pd.DataFrame({
            "coh":['H', 'M', 'L', 'Z'], 
            "acc":[data['H_acc'][0], data['M_acc'][0], data['L_acc'][0], data['Z_acc'][0]]
                                })
        all_acc_df = pd.concat([all_acc_df, acc_df], ignore_index=True)
    
    plot_test_acc(all_acc_df, f_dir, 0.02)
    acc_mean = all_acc_df.groupby('coh').mean()
    acc_std = all_acc_df.groupby('coh').std()
    acc_mean.to_csv(join(f_dir, 'test_acc_mean.csv'))
    acc_std.to_csv(join(f_dir, 'test_acc_std.csv'))
            

def plot_test_acc(acc_df, f_dir, lr, rep=None):
    if rep is not None:
        sns.barplot(data = acc_df, x='coh', y='acc', ci="sd").set_title("Test accuracy (Learning Rate = %.2f Rep %d), Errorbar=std" % (lr, rep))
        plt.ylim([0, 1])
        plt.savefig(join(f_dir, "test_acc_lr%f_rep%d.pdf" % (lr, rep)), format="pdf")
        plt.close()
    else:
        sns.barplot(data = acc_df, x='coh', y='acc', ci="sd").set_title("All test accuracy (Learning Rate = %.2f), Errorbar=std" % lr)
        plt.ylim([0, 1])
        plt.savefig(join(f_dir, "test_acc_lr%f_total.pdf" % lr), format="pdf")
        plt.close()

if __name__ == "__main__":
    if plot_shuf:
        shuf_main()
    else:
        no_shuf_main()