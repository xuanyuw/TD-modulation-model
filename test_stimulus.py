from stimulus import Stimulus
from parameters import par, calc_parameters

calc_parameters()
stim = Stimulus()
trial_info = stim.generate_trial()
stim.plot_neural_input(trial_info)
