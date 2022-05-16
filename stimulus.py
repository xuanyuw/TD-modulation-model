import numpy as np
import matplotlib.pyplot as plt
from parameters import *


class Stimulus:
    def __init__(self):
        self.green_dirs = np.array([55, 75, 135, 195, 215])
        self.red_dirs = np.array([235, 255, 315, 15, 35])
        # randomly choose trials with pure noise

    def generate_trial(self):
        self.pure_noise = np.random.choice([0, 1], size=(par['batch_size'],),
                                           p=[(1-par['pure_noise_perc']), par['pure_noise_perc']])
        if par['trial_type'] == 'MDC':
            # generate tuning functions
            self.move_dirs = np.array(
                [55, 75, 135, 195, 215, 235, 255, 315, 15, 35])
            self.num_motion_dirs = par['num_motion_dirs_MDC']
            self.all_motion_tunings, self.fix_tuning = self.create_tuning_functions(
                par['coherence_levels'])
            trial_info = self.generate_MDC_trial()

        elif par['trial_type'] == 'MDD':
            self.move_dirs = np.array([135, 315])
            self.num_motion_dirs = par['num_motion_dirs_MDD']
            self.all_motion_tunings, self.fix_tuning = self.create_tuning_functions(
                par['coherence_levels'])
            trial_info = self.generate_MDD_trial()
        # input activity needs to be non-negative
        trial_info['neural_input'] = np.maximum(0., trial_info['neural_input'])
        return trial_info

    def generate_MDC_trial(self, for_mdd=False, coh_levels=None, test_mode = False):
        """
        Generate motion-direction categorization (MDC) trial

        Trial outline
        1. Fixation = 500ms
        2. Target = 400ms
        3. Motion Stimulus onset & Saccade to corresponding color, hold = 100ms

        Input:
        Direction: 10 directions (55°, 75°, 135°, 195°, 215°, 235°, 255°, 315°, 15°, 35°)
        Color: 2 target colors (red, green)
        """
        trial_info = {'trial_type': 'MDC',
                      'desired_output':  np.zeros((par['batch_size'], par['num_time_steps'], par['n_choices']), dtype=np.float32),
                      'train_mask':  np.ones((par['batch_size'], par['num_time_steps']), dtype=np.float32),
                      'stimulus':  np.zeros((par['batch_size']), dtype=np.int8),
                      'left':  np.zeros((par['batch_size']), dtype=np.int8),
                      'right':  np.zeros((par['batch_size']), dtype=np.int8),
                      # add noise to neural input
                      'neural_input':  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_size'], par['n_input'])).astype(np.float32),
                      # 'neural_input':  np.zeros(shape=(par['num_time_steps'], par['batch_size'], par['n_input'])).astype(np.float32),
                      'stim_level': []
                      }

        # initialize desired_output with all fixation
        #trial_info['desired_output'][:, :, 0] = 1
        fix_time_rng = par['fix_time_rng']
        target_time_rng = par['target_time_rng']
        stim_time_rng = par['stim_time_rng']


        self.color_rf, self.color_tuning = self.create_color_tuning()
        for t in range(par['batch_size']):
            ####################
            # Generate trial paramaters
            ####################

            if self.pure_noise[t] == 1:
                trial_info['stim_level'].append('N')
                stim_cat = np.random.choice([0, 1])

            else:
                # randomly generate stimulus direction and color in RF for each trial
                stim_dir_ind = np.random.randint(self.num_motion_dirs)

                if self.move_dirs[stim_dir_ind] in [35, 235, 215, 55]:
                    trial_info['stim_level'].append('L')
                elif self.move_dirs[stim_dir_ind] in [75, 195, 15, 255]:
                    trial_info['stim_level'].append('M')
                else:
                    trial_info['stim_level'].append('H')

                if self.move_dirs[stim_dir_ind] in self.red_dirs:
                    stim_cat = 0  # need to choose where the red is
                else:
                    stim_cat = 1  # need to choose green
                if coh_levels is None:
                    trial_info['neural_input'][stim_time_rng, t,
                                               :] += np.reshape(self.all_motion_tunings[0][:, stim_dir_ind], (1, -1))
                else:
                    tuning_idx = par['coherence_levels'].index(coh_levels[t])
                    trial_info['neural_input'][stim_time_rng,
                                               t, :] += np.reshape(self.all_motion_tunings[tuning_idx][:, stim_dir_ind], (1, -1))
                # add motion stimulus noise
                trial_info['neural_input'][stim_time_rng, t, :] += np.random.normal(
                    par['input_mean'], par['stim_noise_sd'], size=(len(stim_time_rng), par['n_input']))

            trial_info['stimulus'][t] = stim_cat
            # set the mask equal to zero during the fixation and target display time
            trial_info['train_mask'][t, np.hstack(
                [fix_time_rng, target_time_rng])] = 0
            # can use a greater weight for test period if needed
            trial_info['train_mask'][t,
                                     stim_time_rng] *= par['test_cost_multiplier']

            if test_mode: # alternate color trial by trial when testing
                self.color_rf, self.color_tuning = self.create_color_tuning()
            if par['num_color_tuned'] > 0:
                # targets are still on screen during stimulus period
                trial_info['neural_input'][np.hstack(
                    [target_time_rng, stim_time_rng]), t, :] \
                    += np.reshape(self.color_tuning[:, 0], (-1, 1)).T

            if par['num_fix_tuned'] > 0:
                # fixation is always on screen
                trial_info['neural_input'][np.hstack(
                    [fix_time_rng, target_time_rng, stim_time_rng]), t, :] += np.reshape(self.fix_tuning[:, 0], (-1, 1)).T

            # get the rf of the color [red_rf, green_rf], 1=left, 2=right
            choice = self.color_rf[stim_cat]
            if choice == 1:
                trial_info['left'][t] = 1
            else:
                trial_info['right'][t] = 1
            # generate desired output
            trial_info['desired_output'][t, stim_time_rng, choice-1] = 1
        return trial_info

    def generate_MDD_trial(self):
        """
        input: 
            3 coherence level: 13%, 25%, 50%
            2 directions: 135°, 315°

        output: 2 target, choose one
        """
        zero_coh = False
        temp_coh = par['coherence_levels'][:]
        if 0 in par['coherence_levels']:
            zero_coh = True
            temp_coh.remove(0)
        coh_levels = np.random.choice(
            temp_coh, size=(1, par['batch_size'],))
        # if zero_coh:
        #     coh_levels[:, np.where(self.pure_noise == 1)[0]] = 0

        trial_info = self.generate_MDC_trial(
            for_mdd=True, coh_levels=coh_levels[0])
        # add random noise to represent different coherence level
        # change noise level for every trial in a batch
        trial_info['trial_type'] = 'MDD'

        n_level = 1-coh_levels
        trial_info['stim_level'] = ['H' if x == np.max(
            coh_levels) else 'L' if x == np.min(coh_levels) else 'M' for x in coh_levels[0]]
        if zero_coh:
            trial_info['stim_level'] = ['N' if self.pure_noise[i] ==
                                        1 else trial_info['stim_level'][i] for i in range(par['batch_size'])]

        # # set the noise level of the pure noise trial to 1
        # n_level[:, np.where(self.pure_noise == 1)] = 1
        # n_level = np.repeat(n_level, len(par['stim_time_rng']), axis=0)
        # n_level = np.repeat(n_level[:, :, np.newaxis], par['n_input'], axis=2)
        # n_level_multiplied = n_level*par['n_level_scale']
        # #noise = np.random.normal(scale = n_level_multiplied, size = (len(par['stim_time_rng']), par['batch_size'], par['n_input']))*n_level_multiplied

        # signal = trial_info['neural_input'][par['stim_time_rng'], :, :]
        # noise = np.random.normal(scale=n_level, size=(
        #     len(par['stim_time_rng']), par['batch_size'], par['n_input']))*n_level_multiplied
        # # np.multiply(np.multiply(signal, noise), n_level)
        # trial_info['neural_input'][par['stim_time_rng'], :, :] = signal + noise

        return trial_info

    def create_tuning_functions(self, motion_tuning_heights):
        """
        Generate tuning functions for the Postle task
        """
        all_motion_tunings = []

        fix_tuning = np.zeros((par['n_input'], 1))

        assert par['num_targets'] == 2, "Only suppport two targets in MDC and MDD tasks"

        # generate list of prefered directions
        pref_dirs = np.arange(0, 360, 360//par['num_motion_tuned'])

        # generate list of possible stimulus directions
        stim_dirs = self.move_dirs
        for coh in motion_tuning_heights:
            motion_tuning = np.zeros((par['n_input'], self.num_motion_dirs))
            for n in range(par['num_motion_tuned']):
                for i in range(len(stim_dirs)):
                    d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                    motion_tuning[n, i] = coh * \
                        np.exp(par['kappa']*d)/np.exp(par['kappa'])
            all_motion_tunings.append(motion_tuning)

        for n in range(par['num_fix_tuned']):
            fix_tuning[par['num_motion_tuned'] +
                       par['num_color_tuned']+n, 0] = par['tuning_height']

        return all_motion_tunings, fix_tuning

    def create_color_tuning(self):

        color_tuning = np.zeros((par['n_input'], 1))
        # generate random receptive field (target locaiton) for each color
        # the first receptive field only receives motion tuning
        #color_rf = [1, 2]
        # [red_rf, green_rf], randomize color projection to RF
        color_rf = np.random.choice([1, 2], size=2, replace=False)
        # divide color tuning neurons into two RFs
        for n in range(par['num_color_tuned']//(par['num_receptive_fields']-1)):
            for i in range(len(color_rf)):
                n_ind = n+(color_rf[i]-1)*(par['num_color_tuned'] //
                                           (par['num_receptive_fields']-1))
                color_tuning[par['num_motion_tuned']+n_ind,
                             0] = par['tuning_height']
        return color_rf, color_tuning

    # def plot_neural_input(self, trial_info):

    #     print(trial_info['desired_output'][:, 0, :].T)
    #     f = plt.figure(figsize=(8, 4))
    #     ax = f.add_subplot(1, 1, 1)
    #     t = np.arange(0, par['time_fixation']+par['time_target']+par['time_stim'], par['dt'])
    #     t -= par['time_fixation']+par['time_target']
    #     t0, t1, t2= np.where(
    #         t == -(par['time_fixation']+par['time_target'])), \
    #         np.where(t == -par['time_target']), \
    #         np.where(t == 0)
    #     im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
    #     im = ax.imshow(trial_info['neural_input'][:, :, 0],
    #                    aspect='auto', interpolation='none')
    #     plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
    #     ax.set_xticks([t0[0], t1[0], t2[0]])
    #     ax.set_xticklabels([-500, 0, 500, 1500])
    #     ax.set_yticks([0, 9, 18, 27])
    #     ax.set_yticklabels([0, 90, 180, 270])
    #     f.colorbar(im, orientation='vertical')
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.set_ylabel('Motion direction')
    #     ax.set_xlabel('Time relative to sample onset (ms)')
    #     ax.set_title('Motion input')
    #     plt.show()
    #     plt.savefig('stimulus.pdf', format='pdf')
