import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import join
import pandas as pd
from pickle import load

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_noFeedback_model"
model_type = f_dir.split('_')[-2]
total_rep = 50
total_shuf = 100
all_lr = [2e-2]
plot_shuf = False
rerun_calculation = False


def shuf_main():
    if rerun_calculation:
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
            plot_test_acc(rep_acc_df, f_dir, rep)
        
        acc_mean = all_acc_df.groupby('coh').mean()
        acc_std = all_acc_df.groupby('coh').std()
        acc_mean.to_csv(join(f_dir, 'test_acc_mean.csv'))
        acc_std.to_csv(join(f_dir, 'test_acc_std.csv'))
        all_acc_df.to_csv(join(f_dir, 'all_test_acc.csv'))
    else:
        all_acc_df = pd.read_csv(join(f_dir, 'all_test_acc.csv'))
    plot_test_acc(all_acc_df, f_dir, model_type)
    
            

def no_shuf_main():
    if rerun_calculation:
        all_acc_df = pd.DataFrame(columns=['coh', 'acc'])
        for rep in range(total_rep):
            with open(join(f_dir, 'test_results_%d.pkl' %(rep)), 'rb') as f:
                data = load(f)
            acc_df = pd.DataFrame({
                "coh":['H', 'M', 'L', 'Z'], 
                "acc":[data['H_acc'][0], data['M_acc'][0], data['L_acc'][0], data['Z_acc'][0]]
                                    })
            all_acc_df = pd.concat([all_acc_df, acc_df], ignore_index=True)
        acc_mean = all_acc_df.groupby('coh').mean()
        acc_std = all_acc_df.groupby('coh').std()
        acc_mean.to_csv(join(f_dir, 'test_acc_mean.csv'))
        acc_std.to_csv(join(f_dir, 'test_acc_std.csv'))
        all_acc_df.to_csv(join(f_dir, 'all_test_acc.csv'))
    else:
        all_acc_df = pd.read_csv(join(f_dir, 'all_test_acc.csv'))
    plot_test_acc(all_acc_df, f_dir, model_type)
    
            

def plot_test_acc(acc_df, f_dir, model_type, rep=None):
    if rep is not None:
        sns.barplot(data = acc_df, x='coh', y='acc', ci="sd").set_title("%s, Test accuracy (Rep %d), Errorbar=std" % (model_type, rep))
        plt.ylim([0, 1])
        plt.savefig(join(f_dir, "test_acc_%s_rep%d.pdf" % (model_type, rep)), format="pdf")
        plt.close()
    else:
        sns.barplot(data = acc_df, x='coh', y='acc', ci="sd").set_title("%s\nAll test accuracy Errorbar=std" % (model_type))
        plt.ylim([0, 1])
        plt.savefig(join(f_dir, "test_acc_%s.pdf" % model_type), format="pdf")
        plt.close()

if __name__ == "__main__":
    if plot_shuf:
        shuf_main()
    else:
        no_shuf_main()