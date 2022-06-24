from glob import has_magic
from model import Model
from stimulus import Stimulus
from analysis import get_perf, get_reaction_time
from os.path import join
import tables
from pickle import dump
from numpy import save, mean, where, array
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


bp.math.set_platform("cpu")


def trial(par, train=True):
    num_iterations = par["num_iterations"]
    iter_between_outputs = par["iters_between_outputs"]
    stim = Stimulus(par)

    model = Model(par, train=train)
    opt = bp.optimizers.Adam(par["learning_rate"], train_vars=model.train_vars())
    if train:
        grad_f = bm.grad(
            model.loss_func,
            dyn_vars=model.vars(),
            grad_vars=model.train_vars(),
            return_value=True,
        )

    @bm.jit
    @bp.math.function(nodes=(model, opt))
    def train_op(x, y, mask):
        grads, _ = grad_f(x, y, mask)
        capped_gs = dict()
        for key, grad in grads.items():
            if "w_rnn" in key:
                grad *= model.rnn_mask
            elif "w_out" in key:
                grad *= model.out_mask
            elif "w_in" in key:
                grad *= model.in_mask
            capped_gs[key] = bm.clip_by_norm(grad, par["clip_max_grad_val"])
        opt.update(grads=capped_gs)

    # keep track of the model performance across training
    if train:
        model_performance = {
            "total_accuracy": [],
            "rt": [],
            "loss": [],
            "perf_loss": [],
            "spike_loss": [],
            "weight_loss": [],
            "iteration": [],
        }
    else:
        model_performance = {
            "total_accuracy": [],
            "rt": [],
            "loss": [],
            "perf_loss": [],
            "spike_loss": [],
            "weight_loss": [],
            "iteration": [],
            "H_acc": [],
            "M_acc": [],
            "L_acc": [],
            "Z_acc": [],
        }

    if par["save_train_out"]:
        all_y_hist = []
        all_target = []
        all_stim_level = []
        all_stim_dir = []
        all_h = []
        all_rt = []
        all_neural_in = []
        all_in_weight = []
        all_rnn_weight = []
        all_out_weight = []
        all_b_out = []
        all_b_rnn = []
        all_idx = []
    for i in range(num_iterations):
        model.reset()
        # generate batch of batch_train_size
        stim = Stimulus(par)
        trial_info = stim.generate_trial()
        inputs = bm.array(trial_info["neural_input"], dtype=bm.float32)
        targets = bm.array(trial_info["desired_output"], dtype=bm.float32)
        mask = bm.array(trial_info["train_mask"], dtype=bm.float32)

        # Run the model
        if train:
            train_op(inputs, targets, mask)
            # get metrics
            _, total_accuracy = get_perf(
                targets, model.y_hist, mask, trial_info["stim_level"], True
            )
            rt = mean(get_reaction_time(model.y_hist, par))
            model_performance["total_accuracy"].append(total_accuracy)
            model_performance["rt"].append(rt)
            model_performance["loss"].append(model.loss.numpy()[0])
            model_performance["perf_loss"].append(model.perf_loss.numpy()[0])
            model_performance["spike_loss"].append(model.spike_loss.numpy()[0])
            model_performance["weight_loss"].append(model.weight_loss.numpy()[0])

            # stop training if all trials with meaningful inputs reaches 95% accuracy
            # cond = (array([H_acc, M_acc, L_acc])>0.95).all()
            cond = total_accuracy > 0.95
            # Save the network model and output model performance to screen
            if i % iter_between_outputs == 0 or i == num_iterations - 1 or cond:
                # save training output
                if par["save_train_out"]:
                    all_idx.append(i)
                    all_y_hist.append(model.y_hist.numpy())
                    all_target.append(targets.numpy())
                    all_stim_level.append(trial_info["stim_level"])
                    all_stim_dir.append(trial_info["stim_dir"])
                    all_h.append(model.h_hist.numpy())
                    all_rt.append(get_reaction_time(model.y_hist, par))
                    all_neural_in.append(inputs.numpy())
                    all_in_weight.append(model.w_in.numpy())
                    all_rnn_weight.append(model.w_rnn.numpy())
                    all_out_weight.append(model.w_out.numpy())
                    all_b_out.append(model.b_out.numpy())
                    all_b_rnn.append(model.b_rnn.numpy())

                print(
                    f" Iter {i:4d}"
                    + f" | Accuracy {total_accuracy:0.4f}"
                    + f" | Perf loss {model.perf_loss[0]:0.4f}"
                    + f" | Spike loss {model.spike_loss[0]:0.4f}"
                    + f" | Weight loss {model.weight_loss[0]:0.4f}"
                    + f" | Mean activity {bm.mean(model.h):0.4f}"
                    + f" | RT {rt:0.4f}"
                )
                # print(f'Separated Accuracy:' +
                #       f' | H {H_acc:0.4f}' +
                #       f' | M {M_acc:0.4f}' +
                #       f' | L {L_acc:0.4f}' +
                #       f' | Z {Z_acc:0.4f}')
                print(
                    "--------------------------------------------------------------------------------------------------------------------------------"
                )
            if cond:
                break

        else:
            # get metrics
            logits, h_hist = model.predict(inputs, False)
            accuracy, total_accuracy = get_perf(
                targets, logits, mask, trial_info["stim_level"], False
            )
            H_acc = accuracy["H"]
            M_acc = accuracy["M"]
            L_acc = accuracy["L"]
            Z_acc = accuracy["Z"]
            rt = mean(get_reaction_time(model.y_hist, par))
            model_performance["total_accuracy"].append(total_accuracy)
            model_performance["rt"].append(rt)
            model_performance["H_acc"].append(H_acc)
            model_performance["M_acc"].append(M_acc)
            model_performance["L_acc"].append(L_acc)
            model_performance["Z_acc"].append(Z_acc)

            # save test output at every test iteration
            if par["save_test_out"]:
                all_idx.append(i)
                all_y_hist.append(logits.numpy())
                all_target.append(targets.numpy())
                all_stim_level.append(trial_info["stim_level"])
                all_stim_dir.append(trial_info["stim_dir"])
                all_h.append(h_hist.numpy())
                all_rt.append(get_reaction_time(model.y_hist, par))
                all_neural_in.append(inputs.numpy())
            print(
                f" Iter {i:4d}"
                + f" | Accuracy {total_accuracy:0.4f}"
                + f" | Mean activity {bm.mean(model.h):0.4f}"
                + f" | RT {rt:0.4f}"
            )
            print(
                f"Separated Accuracy:"
                + f" | H {H_acc:0.4f}"
                + f" | M {M_acc:0.4f}"
                + f" | L {L_acc:0.4f}"
                + f" | Z {Z_acc:0.4f}"
            )
            print(
                "--------------------------------------------------------------------------------------------------------------------------------"
            )

    if train and par["save_train_out"]:
        train_file = tables.open_file(
            join(
                par["save_dir"],
                "train_output_lr%f_rep%d.h5" % (par["learning_rate"], par["rep"]),
            ),
            mode="w",
            title="Training output",
        )
        for n in range(len(all_idx)):
            train_file.create_array(
                "/", "y_hist_iter{}".format(all_idx[n]), all_y_hist[n]
            )
            train_file.create_array(
                "/", "target_iter{}".format(all_idx[n]), all_target[n]
            )
            train_file.create_array(
                "/", "stim_level_iter{}".format(all_idx[n]), all_stim_level[n]
            )
            train_file.create_array(
                "/", "stim_dir_iter{}".format(all_idx[n]), all_stim_dir[n]
            )
            train_file.create_array("/", "h_iter{}".format(all_idx[n]), all_h[n])
            train_file.create_array("/", "rt_iter{}".format(all_idx[n]), all_rt[n])
            train_file.create_array(
                "/", "neural_in_iter{}".format(all_idx[n]), all_neural_in[n]
            )
            train_file.create_array(
                "/", "w_in_iter{}".format(all_idx[n]), all_in_weight[n]
            )
            train_file.create_array(
                "/", "w_rnn_iter{}".format(all_idx[n]), all_rnn_weight[n]
            )
            train_file.create_array(
                "/", "w_out_iter{}".format(all_idx[n]), all_out_weight[n]
            )
            train_file.create_array(
                "/", "b_out_iter{}".format(all_idx[n]), all_b_out[n]
            )
            train_file.create_array(
                "/", "b_rnn_iter{}".format(all_idx[n]), all_b_rnn[n]
            )
        train_file.close()
    if not train and par["save_test_out"]:
        test_file = tables.open_file(
            join(
                par["save_dir"],
                "test_output_lr%f_rep%d.h5" % (par["learning_rate"], par["rep"]),
            ),
            mode="w",
            title="Test output",
        )
        for n in range(len(all_idx)):
            test_file.create_array(
                "/", "y_hist_iter{}".format(all_idx[n]), all_y_hist[n]
            )
            test_file.create_array(
                "/", "target_iter{}".format(all_idx[n]), all_target[n]
            )
            test_file.create_array(
                "/", "stim_level_iter{}".format(all_idx[n]), all_stim_level[n]
            )
            test_file.create_array(
                "/", "stim_dir_iter{}".format(all_idx[n]), all_stim_dir[n]
            )
            test_file.create_array("/", "h_iter{}".format(all_idx[n]), all_h[n])
            test_file.create_array("/", "rt_iter{}".format(all_idx[n]), all_rt[n])
            test_file.create_array(
                "/", "neural_in_iter{}".format(all_idx[n]), all_neural_in[n]
            )
        test_file.close()

    if train:  # plot train accuracy and loss
        plot_acc(
            model_performance["total_accuracy"],
            model_performance["loss"],
            par["save_dir"],
            par["learning_rate"],
            par["rep"],
        )

        # Save model and results
        weights = {}
        w = model.train_vars().unique().dict()
        for k, v in w.items():
            temp = k.split(".")
            weights[temp[1] + "0"] = v

        # Save weight masks
        all_masks = model.get_all_masks()
        for k in all_masks.keys():
            weights[k] = all_masks[k]

        with open(join(par["save_dir"], par["weight_fn"]), "wb") as f:
            save(f, weights)
    else:  # plot test acc
        plot_test_acc(
            [H_acc, M_acc, L_acc, Z_acc],
            par["save_dir"],
            par["learning_rate"],
            par["rep"],
        )
    dump(model_performance, open(join(par["save_dir"], par["save_fn"]), "wb"))


def plot_acc(
    all_arr, loss, f_dir, lr, rep, h_arr=None, m_arr=None, l_arr=None, z_arr=None
):
    f = plt.figure(figsize=(10, 6))
    ax = f.add_subplot(2, 1, 1)
    ax.axhline(y=0.9, color="k", linestyle="--")
    if h_arr is not None:
        ax.plot(h_arr, "#7CB9E8", alpha=0.6, label="high coh")
    if m_arr is not None:
        ax.plot(m_arr, "#007FFF", alpha=0.6, label="mid coh")
    if l_arr is not None:
        ax.plot(l_arr, "#00308F", alpha=0.6, label="low coh")
    if z_arr is not None:
        ax.plot(z_arr, "gray", alpha=0.6, label="zero coh")
    ax.plot(all_arr, "r", linewidth=2, label="total acc")
    ax.legend()
    ax.set_title("Learning Rate = %f, Rep %d" % (lr, rep))
    ax2 = f.add_subplot(2, 1, 2)
    ax2.plot(loss)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, len(loss) - 1)
    ax2.set_title("Loss")
    plt.savefig(join(f_dir, "TrainAcc_lr%f_rep%d.png" % (lr, rep)), format="png")
    # plt.show()


def plot_test_acc(acc_arr, f_dir, lr, rep):
    fig, ax = plt.subplots()
    bars = ax.bar(["H", "M", "L", "Z"], acc_arr)
    ax.set_xlabel("coherence")
    ax.set_ylabel("accuracy")
    ax.set_title("Test accuracy (Learning Rate = %fm Rep %d" % (lr, rep))
    ax.bar_label(bars)
    # plt.show()
    plt.savefig(join(f_dir, "test_acc_lr%f_rep%d.png" % (lr, rep)), format="png")

