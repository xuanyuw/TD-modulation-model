import numpy as np
import matplotlib.pyplot as plt
from os.path import join

model_dir = "trained_withoutFB_model"

conn_num_list = []
for rep in range(50):
    weight_fn = "weight_%d_lr%f.pth" % (rep, 0.02)

    all_weights = np.load(join(model_dir, weight_fn), allow_pickle=True)
    t = all_weights.item()["rnn_mask_init"]
    w = all_weights.item()["w_rnn0"]
    conn_num_list.append(sum(t.flatten()))

plt.hist(conn_num_list)
yl = plt.ylim()
plt.vlines(np.mean(conn_num_list), 0, max(yl), colors="r", linestyles="dashed")
plt.title("Distribution of number of initialized connection")
plt.show()
