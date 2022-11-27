import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pickle import load
import pandas as pd
from os.path import join
import numpy as np
import tables
from utils import find_pref_dir

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
model_type = f_dir.split('_')[-2]
total_rep = 50
total_shuf = 100
lr = 2e-2
plot_sel = True
rerun_calculation = False

def find_targ_color_neuron(mask_in):
    m1_red = np.sum(mask_in[9:11, :], axis=0).astype(bool)
    m1_green = np.sum(mask_in[11:13, :], axis=0).astype(bool)
    m2_red = np.sum(mask_in[13:15, :], axis=0).astype(bool)
    m2_green = np.sum(mask_in[15:17, :], axis=0).astype(bool)
    return m1_red, m1_green, m2_red, m2_green

def locate_neurons(from_arr, to_arr, mask):
    from_m = np.tile(np.expand_dims(from_arr, axis=1), (1, len(from_arr)))
    to_m = np.tile(to_arr, (len(to_arr), 1))
    return from_m * to_m * mask.astype(bool)

def find_weights(from_arr, to_arr, mask, w):
    loc_m = locate_neurons(from_arr, to_arr, mask)
    temp = w * loc_m
    return temp[temp != 0]

def load_data():
    if rerun_calculation:
        df = pd.DataFrame(columns=['conn', 'weights', 'conn_type', 'module', 'rep'])
        for rep in range(total_rep):
            print('Loading rep {}'.format(rep))
            init_w = np.load(join(f_dir, 'init_weight_%d_lr%f.pth'%(rep, lr)), allow_pickle=True)
            init_w = init_w.item()
            trained_w = np.load(join(f_dir, 'weight_%d_lr%f.pth'%(rep, lr)), allow_pickle=True)
            trained_w = trained_w.item()
            m1_red, m1_green, m2_red, m2_green = find_targ_color_neuron(init_w['in_mask_init'])
            test_output = tables.open_file(join(f_dir, 'test_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
            test_table = test_output.root
            pref_red = find_pref_dir(test_table['stim_level_iter0'][:], test_table['stim_dir_iter0'][:], test_table['h_iter0'][:])
            pref_green = ~pref_red.astype(bool)
            m1_pref_red, m2_pref_red, m1_pref_green, m2_pref_green  = np.zeros(pref_red.shape), np.zeros(pref_red.shape), np.zeros(pref_red.shape), np.zeros(pref_red.shape)
            m1_pref_red[:50] = pref_red[:50]
            m2_pref_red[100:150] = pref_red[100:150]
            m1_pref_green[:50] = pref_green[:50]
            m2_pref_green[100:150] = pref_green[100:150]

            trained_w_rnn =  trained_w['w_rnn0']
            rnn_mask = trained_w['rnn_mask_init']

            # feedforward conn
            m1_mr2tr = find_weights(m1_pref_red, m1_red, rnn_mask, trained_w_rnn)
            m1_mg2tr = find_weights(m1_pref_green, m1_red, rnn_mask, trained_w_rnn)
            m1_mr2tg = find_weights(m1_pref_red, m1_green, rnn_mask, trained_w_rnn)
            m1_mg2tg = find_weights(m1_pref_green, m1_green, rnn_mask, trained_w_rnn)
            m2_mr2tr = find_weights(m2_pref_red, m2_red, rnn_mask, trained_w_rnn)
            m2_mg2tr = find_weights(m2_pref_green, m2_red, rnn_mask, trained_w_rnn)
            m2_mr2tg = find_weights(m2_pref_red, m1_green, rnn_mask, trained_w_rnn)
            m2_mg2tg = find_weights(m2_pref_green, m1_green, rnn_mask, trained_w_rnn)

            #feedback conn
            m1_tr2mr = find_weights(m1_red, m1_pref_red, rnn_mask, trained_w_rnn)
            m1_tr2mg = find_weights(m1_red, m1_pref_green, rnn_mask, trained_w_rnn)
            m1_tg2mr = find_weights(m1_green, m1_pref_red, rnn_mask, trained_w_rnn)
            m1_tg2mg = find_weights(m1_green, m1_pref_green, rnn_mask, trained_w_rnn)
            m2_tr2mr = find_weights(m2_red, m2_pref_red, rnn_mask, trained_w_rnn)
            m2_tr2mg = find_weights(m2_red, m2_pref_green, rnn_mask, trained_w_rnn)
            m2_tg2mr = find_weights(m2_green, m2_pref_red, rnn_mask, trained_w_rnn)
            m2_tg2mg = find_weights(m2_green, m2_pref_green, rnn_mask, trained_w_rnn)

            conn = ['mr-tr', 'mg-tr', 'mr-tg', 'mg-tg'] * 4
            conn_type = ['ff'] * 8 + ['fb'] * 8
            module = ['m1'] * 4 + ['m2'] * 4 + ['m1'] * 4 + ['m2'] * 4
            rep_num = [rep] * 16
            w_arr = [m1_mr2tr, m1_mg2tr, m1_mr2tg, m1_mg2tg, m2_mr2tr, m2_mg2tr, m2_mr2tg, m2_mg2tg, m1_tr2mr, m1_tr2mg, m1_tg2mr, m1_tg2mg, m2_tr2mr, m2_tr2mg, m2_tg2mr, m2_tg2mg]

            temp_df = pd.DataFrame({'conn': conn, 'weights': w_arr, 'conn_type': conn_type, 'module': module, 'rep': rep_num})
            temp_df = temp_df.explode('weights').reset_index(drop=True)

            df = pd.concat([df, temp_df])
        df.to_csv(join(f_dir, '%s_connections.csv'%model_type))
    else:
        df = pd.read_csv(join(f_dir, '%s_connections.csv'%model_type))
    return df
   
def plot_w_distr(df, rep=None):
    fig, ax = plt.subplots(figsize=(5, 8))
    sns.catplot(x="conn", y="weights",
            hue="conn_type",
            data=df, kind='bar',
            palette="dark", alpha=.6)
    if rep is None:
        title = "%s weight comparison (all)"%model_type
        fn = "%s_weight_comparison.pdf"%model_type
    else:
        title = "%s weight comparison (rep %d)"%(model_type, rep)
        fn = "%s_weight_comparison_%d.pdf"%(model_type, rep)
    plt.savefig(join(f_dir, fn))
    plt.close(fig)

def main():
    df = load_data()
    for rep in range(total_rep):
        temp_df = df.loc[df['rep']==rep]
        plot_w_distr(temp_df, rep)
    plot_w_distr(df)
   

        

if __name__ == '__main__':
    main()
    
    