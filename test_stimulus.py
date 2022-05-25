from stimulus import Stimulus
from calc_params import par

stim = Stimulus(par)
trial_info = stim.generate_trial()
stim.plot_neural_input(trial_info)
