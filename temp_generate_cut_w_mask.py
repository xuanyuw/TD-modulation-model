import numpy as np
import matplotlib.pyplot as plt
from init_weight import fill_mask
from utils import calculate_rf_rngs
import brainpy.math as bm

par = {
    "n_hidden": 200,
    "within_rf_conn_prob": 0.5 * 1.176,
    "cross_rf_conn_prob": 0.25 * 1.176,
    "cross_module_conn_prob": 0.1 * 1.176,
}


def generate_cut_mask():
    rnn_mask_init = np.zeros((par["n_hidden"], par["n_hidden"]))
    h_prob, m_prob, l_prob = (
        par["within_rf_conn_prob"],
        par["cross_rf_conn_prob"],
        par["cross_module_conn_prob"],
    )
    temp_probs = np.array(
        [
            [h_prob, m_prob, l_prob, 0],
            [0, h_prob, 0, l_prob],
            [l_prob, 0, h_prob, m_prob],
            [0, l_prob, 0, h_prob],
        ]
    )
    inh_probs = np.array(
        [
            [h_prob, m_prob, 0, 0],
            [0, h_prob, 0, 0],
            [0, 0, h_prob, m_prob],
            [0, 0, 0, h_prob],
        ]
    )

    # conn_probs = np.tile(temp_probs, (1, 2))
    conn_probs = np.vstack([np.tile(temp_probs, (1, 2)), np.tile(inh_probs, (1, 2))])
    rf_rngs = calculate_rf_rngs()
    rnn_mask_init = fill_mask(rf_rngs, conn_probs, rnn_mask_init)
    # rnn_mask_init = add_interneuron_mask(rnn_mask_init, rf_rngs, l_prob)

    # remove self_connections
    temp_mask = bm.ones((par["n_hidden"], par["n_hidden"])) - bm.eye(par["n_hidden"])
    rnn_mask_init = rnn_mask_init * temp_mask
    return rnn_mask_init


num_conn_list = []
for i in range(50):
    rnn_mask_init = generate_cut_mask()
    num_conn_list.append(sum(rnn_mask_init.flatten()))

plt.hist(num_conn_list)
yl = plt.ylim()
plt.vlines(np.mean(num_conn_list), 0, max(yl), colors="r", linestyles="dashed")
plt.title("Distribution of number of initialized connection")
plt.show()
