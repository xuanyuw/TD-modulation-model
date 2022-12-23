import numpy as np
import matplotlib.pyplot as plt
all_motion_tunings = []
num_motion_tuned = 9
move_dirs = [135, 315]
coh_list = [0, 0.6, 0.9]
kappa = 2
tuning_height = 2
pref_dirs = np.arange(0, 360, 360 // num_motion_tuned)

input_mean = 0
dt = 20
membrane_time_constant = 100
noise_in_sd = 0.07
alpha_neuron = np.float32(dt / (membrane_time_constant))
noise_in = np.sqrt(2 / alpha_neuron) * noise_in_sd
motion_noise_mult = 1

for coh in coh_list:
    motion_tuning = np.zeros((9, len(move_dirs)))
    motion_tuning[: num_motion_tuned, :] = np.random.normal(
                input_mean,
                noise_in * motion_noise_mult,
                size=(num_motion_tuned, len(move_dirs)),
            ).astype(np.float32)
    if coh != 0:
        for n in range(num_motion_tuned):
            for i in range(len(move_dirs)):
                
                d = np.cos((move_dirs[i] - pref_dirs[n]) / 180 * np.pi)
                # motion_tuning[n, i] += np.exp(
                #     self.par['kappa']*d)/np.exp(self.par['kappa'])
                motion_tuning[n, i] += (
                    np.exp(kappa * d)
                    / np.exp(kappa)
                    * coh
                    * tuning_height # increase tuing height for motion input
                )
    else:
        pk = (
            np.exp(kappa)
            / np.exp(kappa) 
            * max(coh_list)
            * tuning_height # increase tuing height for motion input
            )
        motion_tuning += pk * 0.2
    all_motion_tunings.append(motion_tuning)

fig, [ax1, ax2, ax3]  = plt.subplots(1, 3, sharey=True)
ax1.plot(all_motion_tunings[0][:, 0])
ax1.plot(all_motion_tunings[0][:, 1])
ax1.set_title('coh = 0')


ax2.plot(all_motion_tunings[1][:, 0])
ax2.plot(all_motion_tunings[1][:, 1])
ax2.set_title('coh = 0.6')

ax3.plot(all_motion_tunings[2][:, 0])
ax3.plot(all_motion_tunings[2][:, 1])
ax3.set_title('coh = 0.9')
plt.show()


