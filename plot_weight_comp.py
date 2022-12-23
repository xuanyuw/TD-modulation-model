import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pickle import load
import pandas as pd
from os.path import join
import numpy as np
import tables
from utils import find_pref_dir, find_pref_targ_color, pick_selective_neurons, min_max_normalize,recover_targ_loc, relu
from calc_params import par

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
model_type = f_dir.split('_')[-2]
total_rep = 50
total_shuf = 100
lr = 2e-2
plot_sel = True
rerun_calculation = True
plot_trained = False

title = "%s_weight_comparison"%model_type
if plot_sel:
    title += '_selective'
else:
    title += '_allUnits'

if plot_trained:
    title += '_trained'
else:
    title += '_init'

def locate_neurons(from_arr, to_arr, mask):
    from_m = np.tile(np.expand_dims(from_arr, axis=1), (1, len(from_arr)))
    to_m = np.tile(to_arr, (len(to_arr), 1))
    return from_m * to_m * mask.astype(bool)

def find_weights(from_arr, to_arr, mask, w):
    loc_m = locate_neurons(from_arr, to_arr, mask)
    temp = w * loc_m
    return temp[loc_m != 0]

def load_data():
    if rerun_calculation:
        df = pd.DataFrame(columns=['conn', 'weights', 'conn_type', 'module', 'rep'])
        for rep in range(total_rep):
            print('Loading rep {}'.format(rep))
            # laod files
            if plot_trained:
                trained_w = np.load(join(f_dir, 'weight_%d_lr%f.pth'%(rep, lr)), allow_pickle=True)
                trained_w = trained_w.item()
            else:
                init_w = np.load(join(f_dir, 'init_weight_%d_lr%f.pth'%(rep, lr)), allow_pickle=True)
                init_w = init_w.item()
            test_output = tables.open_file(join(f_dir, 'test_output_lr%f_rep%d.h5'%(lr, rep)), mode = 'r')
            test_table = test_output.root
            # find targ color encoding neurons 
            # prefer green target = 0, prefer red target = 1
            m1_targ_rng = np.append(range(40, 80), range(170, 180))
            m2_targ_rng = np.append(range(120, 160), range(190, 200))
            pref_targ_color = find_pref_targ_color(test_table['h_iter0'][:], test_table['target_iter0'][:], test_table['stim_dir_iter0'][:], m1_targ_rng, m2_targ_rng)
            m1_green, m1_red, m2_green, m2_red = np.zeros(pref_targ_color.shape),  np.zeros(pref_targ_color.shape),  np.zeros(pref_targ_color.shape),  np.zeros(pref_targ_color.shape)
            m1_green[m1_targ_rng] = pref_targ_color[m1_targ_rng]==0
            m2_green[m2_targ_rng] = pref_targ_color[m2_targ_rng]==0
            m1_red[m1_targ_rng] = pref_targ_color[m1_targ_rng]==1
            m2_red[m2_targ_rng] = pref_targ_color[m2_targ_rng]==1
            if plot_sel:
                targ_loc = recover_targ_loc(test_table['target_iter0'][:], test_table['stim_dir_iter0'][:])[-1, :]
                normalized_h = min_max_normalize(test_table['h_iter0'][:])
                targ_sel = pick_selective_neurons(normalized_h, targ_loc, window_st=25, window_ed=45, alpha=0.05)
                m1_green = m1_green * targ_sel
                m2_green = m2_green * targ_sel
                m1_red = m1_red * targ_sel
                m2_red = m2_red * targ_sel
            # find moving direction encoding neurons
            m1_stim_rng = np.append(range(0, 40), range(160, 170))
            m2_stim_rng = np.append(range(80, 120), range(180, 190))
            pref_red = find_pref_dir(test_table['stim_level_iter0'][:], test_table['stim_dir_iter0'][:], test_table['h_iter0'][:])
            pref_green = ~pref_red.astype(bool)
            m1_pref_red, m2_pref_red, m1_pref_green, m2_pref_green  = np.zeros(pref_red.shape), np.zeros(pref_red.shape), np.zeros(pref_red.shape), np.zeros(pref_red.shape)
            m1_pref_red[m1_stim_rng] = pref_red[m1_stim_rng]
            m2_pref_red[m2_stim_rng] = pref_red[m2_stim_rng]
            m1_pref_green[m1_stim_rng] = pref_green[m1_stim_rng]
            m2_pref_green[m2_stim_rng] = pref_green[m2_stim_rng]
            if plot_sel:
                stim_sel = pick_selective_neurons(normalized_h, test_table['stim_dir_iter0'][:], alpha=0.05)
                m1_pref_red = m1_pref_red * stim_sel
                m2_pref_red = m2_pref_red * stim_sel
                m1_pref_green = m1_pref_green * stim_sel
                m2_pref_green = m2_pref_green * stim_sel
            if plot_trained:
                w_rnn =  par['EI_matrix'] @ relu(trained_w['w_rnn0'])
                rnn_mask = trained_w['rnn_mask_init']
            else:
                w_rnn = par['EI_matrix'] @ relu(init_w['w_rnn0'])
                rnn_mask = init_w['rnn_mask_init']
                

            # feedforward conn
            m1_mr2tr = find_weights(m1_pref_red, m1_red, rnn_mask, w_rnn)
            m1_mg2tr = find_weights(m1_pref_green, m1_red, rnn_mask, w_rnn)
            m1_mr2tg = find_weights(m1_pref_red, m1_green, rnn_mask, w_rnn)
            m1_mg2tg = find_weights(m1_pref_green, m1_green, rnn_mask, w_rnn)
            m2_mr2tr = find_weights(m2_pref_red, m2_red, rnn_mask, w_rnn)
            m2_mg2tr = find_weights(m2_pref_green, m2_red, rnn_mask, w_rnn)
            m2_mr2tg = find_weights(m2_pref_red, m2_green, rnn_mask, w_rnn)
            m2_mg2tg = find_weights(m2_pref_green, m2_green, rnn_mask, w_rnn)

            #feedback conn
            m1_tr2mr = find_weights(m1_red, m1_pref_red, rnn_mask, w_rnn)
            m1_tr2mg = find_weights(m1_red, m1_pref_green, rnn_mask, w_rnn)
            m1_tg2mr = find_weights(m1_green, m1_pref_red, rnn_mask, w_rnn)
            m1_tg2mg = find_weights(m1_green, m1_pref_green, rnn_mask, w_rnn)
            m2_tr2mr = find_weights(m2_red, m2_pref_red, rnn_mask, w_rnn)
            m2_tr2mg = find_weights(m2_red, m2_pref_green, rnn_mask, w_rnn)
            m2_tg2mr = find_weights(m2_green, m2_pref_red, rnn_mask, w_rnn)
            m2_tg2mg = find_weights(m2_green, m2_pref_green, rnn_mask, w_rnn)

            conn = ['mr-tr', 'mg-tr', 'mr-tg', 'mg-tg'] * 4
            conn_type = ['ff'] * 8 + ['fb'] * 8
            module = ['m1'] * 4 + ['m2'] * 4 + ['m1'] * 4 + ['m2'] * 4
            rep_num = [rep] * 16
            w_arr = [m1_mr2tr, m1_mg2tr, m1_mr2tg, m1_mg2tg, m2_mr2tr, m2_mg2tr, m2_mr2tg, m2_mg2tg, m1_tr2mr, m1_tr2mg, m1_tg2mr, m1_tg2mg, m2_tr2mr, m2_tr2mg, m2_tg2mr, m2_tg2mg]

            temp_df = pd.DataFrame({'conn': conn, 'weights': w_arr, 'conn_type': conn_type, 'module': module, 'rep': rep_num})
            temp_df = temp_df.explode('weights').reset_index(drop=True)

            df = pd.concat([df, temp_df])
        df.to_csv(join(f_dir, title+'_data.csv'))
    else:
        df = pd.read_csv(join(f_dir, title+'_data.csv'))
    return df

   
def plot_w_distr(df, rep=None):

    # if rep is None:
    
    sns.catplot(x="conn", y="weights",
        hue="conn_type",
        data=df, kind='bar',
        palette="dark", alpha=.6)

    fn_png = title + '.png'
    fn_pdf = title + '.pdf'
    # else:
    #     title = "%s weight comparison (rep %d)"%(model_type, rep)
    #     fn = "%s_weight_comparison_%d.pdf"%(model_type, rep)
    plt.title(title)
    plt.savefig(join(f_dir, fn_png), bbox_inches='tight')
    plt.savefig(join(f_dir, fn_pdf), bbox_inches='tight')
    

def main():
    df = load_data()
    # for rep in range(total_rep):
    #     temp_df = df.loc[df['rep']==rep]
    #     plot_w_distr(temp_df, rep)
    plot_w_distr(df)
   

        

if __name__ == '__main__':
    main()
    
    