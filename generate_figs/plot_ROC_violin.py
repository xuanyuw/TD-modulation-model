import sys
import os
# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from plot_ROC import get_sel_cells
import seaborn as sns
from pickle import load
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from statannotations.Annotator import Annotator

# plot settings

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['lines.linewidth'] = 2


f_dir = "F:\Github\TD-modulation-model\crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
total_rep = 50
# f_dir = "/Users/xuanyuwu/Documents/GitHub/TD-modulation-model/crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
all_rep = range(total_rep)
lr = 0.02

stim_st_time = 45
target_st_time = 25
normalize = False
sep_sac = True
plot_sel = True
rerun_calc = False
save_plot = True

# define plot window
st = 29
ed = 48

if not sep_sac:
    plt.rcParams['figure.figsize'] = [6, 4]

# load data
fn = os.path.join(f_dir, 'sep_sac_ROC_dir_%dnet.pkl'%total_rep)
with open(fn, 'rb') as f:
    H_ipsi_dir_ROC, H_contra_dir_ROC, M_ipsi_dir_ROC, M_contra_dir_ROC, L_ipsi_dir_ROC, L_contra_dir_ROC, Z_ipsi_dir_ROC, Z_contra_dir_ROC = load(f)

if rerun_calc:
    motion_selective, _ = get_sel_cells()
    motion_rng = np.concatenate((np.arange(0, 40), np.arange(80, 120), np.arange(160, 170), np.arange(180, 190)))

    H_ipsi_dir_ROC = np.reshape(H_ipsi_dir_ROC, (48, 50, 100))
    H_contra_dir_ROC = np.reshape(H_contra_dir_ROC, (48, 50, 100))
    M_ipsi_dir_ROC = np.reshape(M_ipsi_dir_ROC, (48, 50, 100))
    M_contra_dir_ROC = np.reshape(M_contra_dir_ROC, (48, 50, 100))
    L_ipsi_dir_ROC = np.reshape(L_ipsi_dir_ROC, (48, 50, 100))
    L_contra_dir_ROC = np.reshape(L_contra_dir_ROC, (48, 50, 100))
    Z_ipsi_dir_ROC = np.reshape(Z_ipsi_dir_ROC, (48, 50, 100))
    Z_contra_dir_ROC = np.reshape(Z_contra_dir_ROC, (48, 50, 100))

    H_ipsi_dir_ROC_temp = np.ones(H_ipsi_dir_ROC.shape)*np.nan
    H_contra_dir_ROC_temp = np.ones(H_contra_dir_ROC.shape)*np.nan
    M_ipsi_dir_ROC_temp = np.ones(M_ipsi_dir_ROC.shape)*np.nan
    M_contra_dir_ROC_temp = np.ones(M_contra_dir_ROC.shape)*np.nan
    L_ipsi_dir_ROC_temp = np.ones(L_ipsi_dir_ROC.shape)*np.nan
    L_contra_dir_ROC_temp = np.ones(L_contra_dir_ROC.shape)*np.nan
    Z_ipsi_dir_ROC_temp = np.ones(Z_ipsi_dir_ROC.shape)*np.nan
    Z_contra_dir_ROC_temp = np.ones(Z_contra_dir_ROC.shape)*np.nan


    if plot_sel:
        for rep in all_rep:
            m_sel = motion_selective[rep, motion_rng].astype(bool)

            H_ipsi_dir_ROC_temp[:, rep, :sum(m_sel)] = H_ipsi_dir_ROC[:, rep, m_sel]
            H_contra_dir_ROC_temp[:, rep, :sum(m_sel)] = H_contra_dir_ROC[:, rep, m_sel]
            M_ipsi_dir_ROC_temp[:, rep, :sum(m_sel)] = M_ipsi_dir_ROC[:, rep, m_sel]
            M_contra_dir_ROC_temp[:, rep, :sum(m_sel)] = M_contra_dir_ROC[:, rep, m_sel]
            L_ipsi_dir_ROC_temp[:, rep, :sum(m_sel)] = L_ipsi_dir_ROC[:, rep, m_sel]
            L_contra_dir_ROC_temp[:, rep, :sum(m_sel)] = L_contra_dir_ROC[:, rep, m_sel]
            Z_ipsi_dir_ROC_temp[:, rep, :sum(m_sel)] = Z_ipsi_dir_ROC[:, rep, m_sel]
            Z_contra_dir_ROC_temp[:, rep, :sum(m_sel)] = Z_contra_dir_ROC[:, rep, m_sel]
    else:
        H_ipsi_dir_ROC_temp = H_ipsi_dir_ROC
        H_contra_dir_ROC_temp = H_contra_dir_ROC
        M_ipsi_dir_ROC_temp = M_ipsi_dir_ROC
        M_contra_dir_ROC_temp = M_contra_dir_ROC
        L_ipsi_dir_ROC_temp = L_ipsi_dir_ROC
        L_contra_dir_ROC_temp = L_contra_dir_ROC
        Z_ipsi_dir_ROC_temp = Z_ipsi_dir_ROC
        Z_contra_dir_ROC_temp = Z_contra_dir_ROC


    H_ipsi_dir_netROC =  np.nanmean(H_ipsi_dir_ROC_temp[st:ed, :, :], axis=(0, 2))
    H_contra_dir_netROC =  np.nanmean(H_contra_dir_ROC_temp[st:ed, :, :], axis=(0, 2))
    M_ipsi_dir_netROC =  np.nanmean(M_ipsi_dir_ROC_temp[st:ed, :, :], axis=(0, 2))
    M_contra_dir_netROC =  np.nanmean(M_contra_dir_ROC_temp[st:ed, :, :], axis=(0, 2))
    L_ipsi_dir_netROC =  np.nanmean(L_ipsi_dir_ROC_temp[st:ed, :, :], axis=(0, 2))
    L_contra_dir_netROC =  np.nanmean(L_contra_dir_ROC_temp[st:ed, :, :], axis=(0, 2))
    Z_ipsi_dir_netROC =  np.nanmean(Z_ipsi_dir_ROC_temp[st:ed, :, :], axis=(0, 2))
    Z_contra_dir_netROC =  np.nanmean(Z_contra_dir_ROC_temp[st:ed, :, :], axis=(0, 2))

    rep = list(range(50))*8
    coh = sum([['H']*50*2, ['M']*50*2, ['L']*50*2, ['Z']*50*2], [])
    sac_dir = sum([['ipsi']*50, ['contra']*50], [])*4
    mean_ROC = np.concatenate((H_ipsi_dir_netROC, H_contra_dir_netROC, M_ipsi_dir_netROC, M_contra_dir_netROC, L_ipsi_dir_netROC, L_contra_dir_netROC, Z_ipsi_dir_netROC, Z_contra_dir_netROC))


    df = pd.DataFrame({'rep': rep, 'coh': coh, 'sac_dir': sac_dir, 'mean_ROC':mean_ROC})
    df.to_csv(os.path.join(f_dir, 'ROC_comp_50net.csv'))
else:
    df = pd.read_csv(os.path.join(f_dir, 'ROC_comp_50net.csv'))
    

fig, ax = plt.subplots()
# color_palette = {'H': '#FF0000', 'M': '#00FF00', 'L':'#0000FF', 'Z': 'k'}
colors = ['#FF0000', '#00FF00', '#0000FF', '#424242']
handles = []
sns.violinplot(x = 'coh', y = 'mean_ROC', hue = 'sac_dir', data = df, inner='points', ax=ax,  palette=['.2', '.5'], hue_order=['ipsi', 'contra'])

for ind, violin in enumerate(ax.findobj(PolyCollection)):
    rgb = to_rgb(colors[ind // 2])
    if ind % 2 != 1:
        rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))

ax.legend(handles=[tuple(handles[1::2]), tuple(handles[::2])], labels=df["sac_dir"].astype('category').cat.categories.to_list(),
          handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower left', frameon=False)


# add statistical test results
pairs = [(('H', 'ipsi'), ('H', 'contra')), (('M', 'ipsi'), ('M', 'contra')), (('L', 'ipsi'), ('L', 'contra')), (('Z', 'ipsi'), ('Z', 'contra'))]


annot = Annotator(ax, pairs, data=df, x='coh', y='mean_ROC', hue='sac_dir', order=['H', 'M', 'L', 'Z'])
annot.configure(test='t-test_ind', text_format='star', loc='outside')
annot.apply_and_annotate()


ax.set(xlabel='Coherence', ylabel="Average ROC")
ax.tick_params(bottom=True, left=True)
plt.tight_layout()
if save_plot:
    pic_dir = os.path.join(f_dir, 'ROC_plots_50net')
    plt.savefig(os.path.join(pic_dir, "dir_sel_net_comp.pdf"), format='pdf')
    plt.savefig(os.path.join(pic_dir, "dir_sel_net_comp.eps"), format='eps')
    plt.savefig(os.path.join(pic_dir, "dir_sel_net_comp.png"), format='png')