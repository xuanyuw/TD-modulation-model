import numpy as np
from model.calc_params import par, update_parameters
from model.model_func import trial
from os import makedirs
from os.path import dirname, exists
import time
from gc import collect
from tqdm import tqdm

# For debugging
# from jax.config import config
# config.update("jax_debug_nans", True)


def try_model(par, train):
    try:
        trial(par, train=train)

    except KeyboardInterrupt:
        quit("Quit by KeyboardInterrupt")


tic = time.perf_counter()
# Run models

for rep in np.arange(par["rep"], par["rep_num"]):
    print("Runnning model %d..." % rep)
    # Train model
    update_parameters(
        {
            "rep": rep,
            "save_fn": "model_results_%d_lr%f.pkl" % (rep, lr),
            "batch_size": par["train_batch_size"],
            "num_iterations": par["num_train_iterations"],
            "coherence_levels": par["train_coherence_levels"],
            # "weight_fn": "weight_%d_lr%f.pth" % (rep, lr),
            "weight_fn": "weight_%d.pth" % (rep),
        }
    )
    if not exists(dirname(par["save_dir"])):
        makedirs(dirname(par["save_dir"]))
    # print("Training model %d" % rep)
    # try_model(par, True)

    # Test model
    if par["shuffle_num"] == 0:  # do not shuffle test feedback conn
        if par["cut_spec"] != []:
            update_parameters(
                {
                    "rep": rep,
                    "save_fn": "test_results_%d_cut.pkl" % (rep),
                    "batch_size": par["test_batch_size"],
                    "num_iterations": par["num_test_iterations"],
                    "coherence_levels": par["test_coherence_levels"],
                }
            )
            try_model(par, False)

        else:
            update_parameters(
                {
                    "rep": rep,
                    "save_fn": "test_results_%d.pkl" % (rep),
                    "batch_size": par["test_batch_size"],
                    "num_iterations": par["num_test_iterations"],
                    "coherence_levels": par["test_coherence_levels"],
                }
            )
            try_model(par, False)

    else:
        for shuf_n in range(par["shuffle_num"]):
            update_parameters(
                {
                    "rep": rep,
                    "shuffle": shuf_n,
                    "save_fn": "test_results_%d_shuf%d.pkl" % (rep, shuf_n),
                    "batch_size": par["test_batch_size"],
                    "num_iterations": par["num_test_iterations"],
                    "coherence_levels": par["test_coherence_levels"],
                }
            )
        if shuf_n % 20 == 0:
            print("Testing model %d shuffle # %d" % (rep, shuf_n))
        try_model(par, False)
        collect()
toc = time.perf_counter()

print("elapsed time = %.2f seconds" % (toc - tic))
