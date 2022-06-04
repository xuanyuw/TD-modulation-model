import numpy as np
import matplotlib.pyplot as plt


class Stimulus:

    def __init__(self, par):
        self.par = par

    def generate_trial(self):
        # print('Initializing Stimuli...')
        self.move_dirs = [135, 315]
        self.all_motion_tunings, self.fix_tuning = self.create_tuning_functions(
            self.par['coherence_levels'])
        trial_info = self.generate_MDD_trial()
        # input activity needs to be non-negative
        trial_info['neural_input'] = np.maximum(0., trial_info['neural_input'])
        return trial_info

    def generate_MDD_trial(self):
        """
        Generate motion-direction discrimination (MDD) trial

        trial_info:
            stim_dir: the moving direction of the random dot stimuli
            #  desired_output: correct color choice at every time point (dim3: layer1  = green, layer2 = red)
            desired_output: correct choice location at every time point (dim3: layer1  = contra, layer2 = ipsi)
            targ_loc: the target arrangement (0 = green contralateral, 1 = red contralateral)
            desired_loc: the correct choice location (0 = contra-lateral choice, 1 = ipsi-lateral choice)
            train_mask: the mask determine which self.part of the trial got trained
            coherence: the coherence level of random dots
            stim_level: the coherence level expressed in letters (for analysis)
            neural_input: the simulated neural input for both stimulus and target.
        """
        trial_info = {'desired_output':  np.zeros((self.par['num_time_steps'], self.par['batch_size'], self.par['n_choices']), dtype=np.float32),
                      'train_mask':  np.ones((self.par['num_time_steps'], self.par['batch_size']), dtype=np.float32),
                      # add noise to neural input
                      'neural_input':  np.random.normal(self.par['input_mean'], self.par['noise_in'], size=(self.par['num_time_steps'], self.par['batch_size'], self.par['n_input'])).astype(np.float32),
                      # 'neural_input':  np.zeros(shape=(self.par['num_time_steps'], self.par['batch_size'], self.par['n_input'])).astype(np.float32),
                      }

        fix_time_rng = self.par['fix_time_rng']
        target_time_rng = self.par['target_time_rng']
        stim_time_rng = self.par['stim_time_rng']

        # generate stimulus directions for each trial in batch
        trial_info['stim_dir'] = np.random.choice(
            self.move_dirs, size=(self.par['batch_size'],))

        # # fill in correct color choice according to stimulus directions
        # trial_info['desired_output'][stim_time_rng, np.reshape(np.where(trial_info['stim_dir']
        #                              == 135), (-1, 1)), 0] = 1
        # trial_info['desired_output'][stim_time_rng, np.reshape(np.where(trial_info['stim_dir']
        #                              == 315), (-1, 1)), 1] = 1

        # generate random target arrangement and desired choice location
        trial_info['targ_loc'] = np.random.choice(
            [0, 1], size=(self.par['batch_size'], ))
        temp_stim = trial_info['stim_dir'] == 315
        trial_info['desired_loc'] = np.logical_xor(
            trial_info['targ_loc'], temp_stim).astype(int)

        # generate desired output from desired choice loc
        trial_info['desired_output'][stim_time_rng, np.reshape(np.where(
            trial_info['desired_loc'] == 0), (-1, 1)), 0] = 1
        trial_info['desired_output'][stim_time_rng, np.reshape(np.where(
            trial_info['desired_loc'] == 1), (-1, 1)), 1] = 1
        # generate training mask
        # set the mask equal to zero during the fixation time
        trial_info['train_mask'][np.hstack([fix_time_rng, target_time_rng]), :] = 0
        # can use a greater weight for test period if needed
        trial_info['train_mask'][stim_time_rng, :] *= self.par['test_cost_multiplier']

        # initialize coherences
        trial_info['coherence'] = np.random.choice(
            self.par['coherence_levels'], size=(self.par['batch_size'],))

        trial_info['stim_level'] = ['Z' if x == self.par['coherence_levels'][0] else 'L' if x == self.par['coherence_levels']
                                    [1] else 'M' if x == self.par['coherence_levels'][2] else 'H' for x in trial_info['coherence']]

        for t in range(self.par['batch_size']):

            # randomly generate stimulus direction and color in RF for each trial
            stim_dir_ind = self.move_dirs.index(trial_info['stim_dir'][t])
            tuning_idx = self.par['coherence_levels'].index(
                trial_info['coherence'][t])
            trial_info['neural_input'][stim_time_rng,
                                       t, :] += self.all_motion_tunings[tuning_idx][:, stim_dir_ind]

            color_tuning = self.create_color_tuning(trial_info['targ_loc'][t])
            if self.par['num_color_tuned'] > 0:
                # targets are still on screen during stimulus period
                trial_info['neural_input'][np.hstack(
                    [target_time_rng, stim_time_rng]), t, :] \
                    += color_tuning

            if self.par['num_fix_tuned'] > 0:
                # fixation is always on screen
                trial_info['neural_input'][np.hstack(
                    [fix_time_rng, target_time_rng, stim_time_rng]), t, :] += self.fix_tuning[:, 0]

        return trial_info

    def create_tuning_functions(self, coh_list):
        """
        Generate tuning functions for the Postle task
        """
        all_motion_tunings = []
        fix_tuning = np.zeros((self.par['n_input'], 1))

        # generate list of prefered directions
        pref_dirs = np.arange(0, 360, 360//self.par['num_motion_tuned'])

        # generate list of possible stimulus directions
        for coh in coh_list:
            # initialize with noise
            motion_tuning = self.par['tuning_height'] * (1-coh) * np.random.normal(
                self.par['input_mean'], self.par['noise_in'], size=(self.par['n_input'], len(self.move_dirs))).astype(np.float32)
            if coh != 0:
                for n in range(self.par['num_motion_tuned']):
                    for i in range(len(self.move_dirs)):
                        d = np.cos(
                            (self.move_dirs[i] - pref_dirs[n])/180*np.pi)
                        motion_tuning[n, i] += np.exp(
                            self.par['kappa']*d)/np.exp(self.par['kappa'])
            all_motion_tunings.append(motion_tuning)

        for n in range(self.par['num_fix_tuned']):
            fix_tuning[self.par['num_motion_tuned'] +
                       self.par['num_color_tuned']+n, 0] = self.par['tuning_height']

        return all_motion_tunings, fix_tuning

    def create_color_tuning(self, targ_loc):
        # targ_loc: 0 = green contralateral, 1 = red contralateral
        color_tuning = np.zeros((self.par['n_input'],))
        cell_per_rf = self.par['num_color_tuned']//(
            self.par['num_receptive_fields']-1)
        # create red and green tuning
        green_val = np.zeros((cell_per_rf,))
        green_val[-cell_per_rf//2:] = 1
        red_val = np.zeros((cell_per_rf,))
        red_val[:-cell_per_rf//2] = 1
        if targ_loc:
            # red in contralateral rf
            color_tuning[self.par['num_motion_tuned']
                :self.par['num_motion_tuned']+cell_per_rf] = red_val
            color_tuning[self.par['num_motion_tuned'] +
                         cell_per_rf:self.par['num_motion_tuned']+self.par['num_color_tuned']] = green_val
        else:
            # green in contralateral rf
            color_tuning[self.par['num_motion_tuned']
                :self.par['num_motion_tuned']+cell_per_rf] = green_val
            color_tuning[self.par['num_motion_tuned'] +
                         cell_per_rf:self.par['num_motion_tuned']+self.par['num_color_tuned']] = red_val
        return color_tuning

    def plot_neural_input(self, trial_info):

        print(trial_info['desired_output'][:, 0, :].T)
        f = plt.figure(figsize=(8, 4))
        ax = f.add_subplot(1, 1, 1)
        t = np.arange(self.par['num_time_steps'])
        t0, t1, t2 = 0, \
            min(self.par['target_time_rng']), \
            min(self.par['stim_time_rng'])
        im = ax.imshow(trial_info['neural_input'][:, 0, :].T,
                       aspect='auto', interpolation='none')

        ax.set_xticks([t0, t1, t2])
        ax.set_xticklabels(['-900', '-400', '0'])
        f.colorbar(im, orientation='vertical')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Input Neurons')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.set_title('Neural input')
        plt.savefig('stimulus.pdf', format='pdf')
        plt.show()
