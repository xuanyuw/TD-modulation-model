from stimulus import Stimulus
from calc_params import par
import matplotlib.pyplot as plt
from numpy import where

stim = Stimulus(par)
trial_info = stim.generate_trial()


def plot_neural_input(trial_info, coh_level):
    trial_idx = where(trial_info['coherence'] == coh_level)[0][0]
    print(trial_info['desired_output'][:, trial_idx, :].T)
    f = plt.figure(figsize=(8, 4))
    ax = f.add_subplot(1, 1, 1)
    t0, t1, t2 = 0, \
        min(par['target_time_rng']), \
        min(par['stim_time_rng'])

    im = ax.imshow(trial_info['neural_input'][:, trial_idx, :].T,
                   aspect='auto', interpolation='none')

    ax.set_xticks([t0, t1, t2])
    ax.set_xticklabels(['-900', '-400', '0'])
    f.colorbar(im, orientation='vertical')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Input Neurons')
    ax.set_xlabel('Time relative to sample onset (ms)')
    ax.set_title('Neural input (coherence = %s)'%coh_level)
    plt.savefig('stimulus_coh%s_varmean.png'%(coh_level), format='png')
    plt.show()

# plot_neural_input(trial_info, 0.3)
plot_neural_input(trial_info, 0.6)
plot_neural_input(trial_info, 0.9)