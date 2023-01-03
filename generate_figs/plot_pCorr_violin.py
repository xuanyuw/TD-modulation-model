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
# from statannotations.Annotator import Annotator
from scipy.io import loadmat
from scipy.stats import ttest_ind

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
rerun_calc = True
save_plot = True

# define plot window
st = 29
ed = 48

if rerun_calc:
    ipsi_pCorr_stim = np.zeros((total_rep, ))
    contra_pCorr_stim = np.zeros((total_rep, ))
    ipsi_pCorr_choice = np.zeros((total_rep, ))
    contra_pCorr_choice = np.zeros((total_rep, ))
    for r in all_rep:
        pCorr_results = loadmat(os.path.join(f_dir, 'pCorr_data', 'pCorr_result_rep%d.mat'%(r)))
        ipsi_pCorr_stim[r] = np.mean(pCorr_results['ipsi_pCorr_stim'][st:ed, :])
        contra_pCorr_stim[r] = np.mean(pCorr_results['contra_pCorr_stim'][st:ed, :])
        ipsi_pCorr_choice[r] = np.mean(pCorr_results['ipsi_pCorr_choice'][st:ed, :])
        contra_pCorr_choice[r] = np.mean(pCorr_results['contra_pCorr_choice'][st:ed, :])
    rep = list(range(50))*4
    r_type = sum([['stim']*50*2, ['choice']*50*2], [])
    sac_dir = sum([['ipsi']*50, ['contra']*50], [])*2
    mean_pCorr = np.concatenate((ipsi_pCorr_stim, contra_pCorr_stim, ipsi_pCorr_choice, contra_pCorr_choice))
    
    df = pd.DataFrame({'rep': rep, 'r_type': r_type, 'sac_dir': sac_dir, 'mean_pCorr':mean_pCorr})
    df.to_csv(os.path.join(f_dir, 'pCorr_comp_50net.csv'))
else:
    df = pd.read_csv(os.path.join(f_dir, 'pCorr_comp_50net.csv'))


# calc p-value of difference
ipsi_stim = df.loc[:49]['mean_pCorr'].to_numpy()
contra_stim = df.loc[50:99]['mean_pCorr'].to_numpy()
ipsi_choice = df.loc[100:149]['mean_pCorr'].to_numpy()
contra_choice= df.loc[150:199]['mean_pCorr'].to_numpy()
stim_diff = contra_stim - ipsi_stim
choice_diff = contra_choice - ipsi_choice
p_val = ttest_ind(stim_diff, choice_diff).pvalue
print("Group difference p-val = {:.2e}".format(p_val))



fig, ax = plt.subplots()
# color_palette = {'H': '#FF0000', 'M': '#00FF00', 'L':'#0000FF', 'Z': 'k'}
colors = ['#FF0000','#0000FF']
handles = []
sns.violinplot(x = 'r_type', y = 'mean_pCorr', hue = 'sac_dir', data = df, inner='points', ax=ax,  palette=['.2', '.5'], hue_order=['ipsi', 'contra'])

for ind, violin in enumerate(ax.findobj(PolyCollection)):
    rgb = to_rgb(colors[ind // 2])
    if ind % 2 != 1:
        rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))

ax.legend(handles=[tuple(handles[1::2]), tuple(handles[::2])], labels=df["sac_dir"].astype('category').cat.categories.to_list(),
          handlelength=2, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower left', frameon=False)


# # add statistical test results
# pairs = [(('H', 'ipsi'), ('H', 'contra')), (('M', 'ipsi'), ('M', 'contra')), (('L', 'ipsi'), ('L', 'contra')), (('Z', 'ipsi'), ('Z', 'contra'))]


# annot = Annotator(ax, pairs, data=df, x='coh', y='mean_ROC', hue='sac_dir', order=['H', 'M', 'L', 'Z'])
# annot.configure(test='t-test_ind', text_format='star', loc='outside')
# annot.apply_and_annotate()

ax.set_xticklabels(['R-stimulus', 'R-choice'])
ax.set(ylabel="Average r", xlabel='')
ax.tick_params(bottom=True, left=True)
plt.tight_layout()
if save_plot:
    pic_dir = os.path.join(f_dir, 'pCorr_plots')
    plt.savefig(os.path.join(pic_dir, "pCorr_net_comp.pdf"), format='pdf')
    plt.savefig(os.path.join(pic_dir, "pCorr_net_comp.eps"), format='eps')
    plt.savefig(os.path.join(pic_dir, "pCorr_net_comp.png"), format='png')