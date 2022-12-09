import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pickle import load
from calc_params import par

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

f_dir = "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model"
model_type = f_dir.split('_')[-2]
total_rep = 50
total_shuf = 100
all_lr = [2e-2]
plot_sel = True


STIM_ST_TIME = (par['time_fixation'] + par['time_target'])//par['dt']
TARG_ST_TIME = par['time_fixation']//par['dt']
DT = par['dt']

def main(lr, total_rep):
    dir_sel_norm, sac_sel_pvnp_norm, sac_sel_lvr_norm = load(open(os.path.join(f_dir, "example_selectivity_data_normalized.pkl"), 'rb'))
    dir_sel_norm = load_new_line_dict(dir_sel_norm)
    sac_sel_pvnp_norm = load_new_line_dict(sac_sel_pvnp_norm)
    sac_sel_lvr_norm = load_new_line_dict(sac_sel_lvr_norm)

    dir_sel_orig, sac_sel_pvnp_orig, sac_sel_lvr_orig = load(open(os.path.join(f_dir, "example_selectivity_data_raw.pkl"), 'rb'))
    dir_sel_orig = load_new_line_dict(dir_sel_orig)
    sac_sel_pvnp_orig = load_new_line_dict(sac_sel_pvnp_orig)
    sac_sel_lvr_orig = load_new_line_dict(sac_sel_lvr_orig)


    plot_total_dir_selectivity(dir_sel_norm, "%s_Motion_dir_sel_norm_avg_combined_example"%model_type, True, plot_sel=plot_sel)
    plot_total_dir_selectivity(dir_sel_orig, "%s_Motion_dir_sel_raw_avg_combined_example"%model_type, True, plot_sel=plot_sel)
    
    plot_total_sac_selectivity_pvnp(sac_sel_pvnp_norm, "%s_Target_sac_sel_pvnp_norm_avg_combined_example"%model_type, True, plot_sel=plot_sel)
    plot_total_sac_selectivity_pvnp(sac_sel_pvnp_orig, "%s_Target_sac_sel_pvnp_raw_avg_combined_example"%model_type, True, plot_sel=plot_sel)

    plot_total_sac_selectivity_lvr(sac_sel_lvr_norm, "%s_Target_sac_sel_lvr_norm_avg_combined_example"%model_type, True, plot_sel=plot_sel)
    plot_total_sac_selectivity_lvr(sac_sel_lvr_orig, "%s_Target_sac_sel_lvr_raw_avg_combined_example"%model_type, True, plot_sel=plot_sel)

def load_new_line_dict(old_dict):
    new_dict = {}
    coh_list = ['H', 'M', 'L', 'Z']
    line_list = ['solid', 'dash']
    for coh in coh_list:
        for l in line_list:
            new_dict['%s_%s'%(coh, l)] = np.mean(np.vstack([old_dict['%s_%s_ax1'%(coh, l)], old_dict['%s_%s_ax2'%(coh, l)]]), axis=0)
    return new_dict

def plot_coh_popu_act(
    line_dict,
    label_dict,
    coh_levels,
    color_dict={"Z": "k", "L": "b", "M": "g", "H": "r"},
    target_st_time=TARG_ST_TIME,
    stim_st_time=STIM_ST_TIME,
):

    assert all(
        k in ["dash", "solid", "ax1_title", "ax2_title", "sup_title"]
        for k in label_dict.keys()
    )
    assert len(line_dict) % 2 == 0 and len(line_dict) >= 2
    assert all("_dash" in k or "_solid" in k for k in line_dict.keys())
    
    fig = plt.figure()
    for i in range(len(coh_levels)):
        if coh_levels[i] + "_dash" not in line_dict.keys():
            continue
        plt.plot(
            line_dict[coh_levels[i] + "_dash"],
            linestyle="--",
            color=color_dict[coh_levels[i]],
            label=coh_levels[i] + "," + label_dict["dash"],
        )
        plt.plot(
            line_dict[coh_levels[i] + "_solid"],
            color=color_dict[coh_levels[i]],
            label=coh_levels[i] + "," + label_dict["solid"],
        )
    yl = plt.ylim()
    
    plt.title(label_dict["sup_title"])
    plt.ylabel("Average activity")
    plt.xlabel("Time")
    xticks = np.array([0, target_st_time, stim_st_time, len(line_dict[coh_levels[i] + "_solid"])])
    plt.xticks(xticks, labels=(xticks-stim_st_time)*DT)
    plt.vlines([target_st_time, stim_st_time], yl[0], yl[1], color="k")
    plt.ylim(yl)

    return fig

def plot_total_dir_selectivity(line_dict, title, save_plt, plot_sel=False):

    label_dict = {
        "dash": "nonPref",
        "solid": "Pref",
        "sup_title": title,
    }

    fig = plot_coh_popu_act(line_dict, label_dict, ['Z', 'L', 'M', 'H'])
    if save_plt:
        folder_n = "popu_act"
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg_lr%f" % (folder_n, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.pdf" % title))
        plt.close(fig)


def plot_total_sac_selectivity_pvnp(line_dict, title, save_plt, plot_sel=False):

    label_dict = {
        "dash": "nonPref",
        "solid": "Pref",
        "sup_title": title,
    }

    fig = plot_coh_popu_act(line_dict, label_dict, ['Z', 'L', 'M', 'H'])
    if save_plt:
        folder_n = "popu_act"
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg_lr%f" % (folder_n, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.pdf" % title))
        plt.close(fig)


def plot_total_sac_selectivity_lvr(line_dict, title, save_plt, plot_sel=False):
    label_dict = {
        "dash": "right",
        "solid": "left",
        "ax1_title": "Module 1",
        "ax2_title": "Module 2",
        "sup_title": title,
    }

    fig = plot_coh_popu_act(line_dict, label_dict, ['Z', 'L', 'M', 'H'])
    if save_plt:
        folder_n = "popu_act"
        if plot_sel:
            folder_n += "_selected"
        pic_dir = os.path.join(f_dir, "%s_avg_lr%f" % (folder_n, lr))
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir, "%s.pdf" % title))
        plt.close(fig)


for lr in all_lr:
    main(lr, total_rep)
