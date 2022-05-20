from calc_params import par
from init_weight import all_weights
import brainpy as bp
import brainpy.math as bm
bp.math.set_platform('cpu')


class Model(bp.layers.Module):

    def __init__(self, par=par, train=True):
        super(Model, self).__init__()

        self.syn_x = bm.Variable(par['syn_x_init'])
        self.syn_u = bm.Variable(par['syn_u_init'])
        self.u = bm.Variable(par['U'])
        self.h = bm.Variable(par['x0'])
        self.init_h = bm.TrainVar(par['x0'])
        self.alpha_std = bm.array(par['alpha_std'])
        self.alpha_stf = bm.array(par['alpha_stf'])
        self.dynamic_synapse = bm.array(par['dynamic_synapse'])
        self.alpha = bm.array(par['alpha_neuron'])

        self.in_mask = bm.array(all_weights['in_mask_init'])
        self.rnn_mask = bm.array(all_weights['rnn_mask_init'])
        self.out_mask = bm.array(all_weights['out_mask_init'])
        self.w_in = bm.TrainVar(all_weights['w_in0'])
        self.w_rnn = bm.TrainVar(all_weights['w_rnn_base0'])
        self.w_out = bm.TrainVar(all_weights['w_out0'])
        self.b_rnn = bm.TrainVar(all_weights['b_rnn0'])
        self.b_out = bm.TrainVar(all_weights['b_out0'])

        self.y = bm.Variable(bm.ones((par['batch_size'], par['n_output'])))
        self.y_hist = bm.Variable(
            bm.zeros((par['num_time_steps'], par['batch_size'], par['n_output'])))

        # Loss
        self.loss = bm.Variable(bm.zeros(1))
        self.perf_loss = bm.Variable(bm.zeros(1))
        self.spike_loss = bm.Variable(bm.zeros(1))
        self.weight_loss = bm.Variable(bm.zeros(1))

        if not train:
            state_dict = np.load(par['save_dir'] + par['weight_fn'])
            state_dict = state_dict.items()
            for k, v in state_dict.items():
                self[k] = v

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __getitem__(self, k):
        getattr(self, k)

    def reset(self):

        self.u.value = bm.array(par['U'])
        self.syn_x.value = bm.array(par['syn_x_init'])
        self.syn_u.value = bm.array(par['syn_u_init'])
        self.loss[:] = 0.
        self.perf_loss[:] = 0.
        self.spike_loss[:] = 0.
        self.weight_loss[:] = 0.

    def update(self, input, **kwargs):
        if par['synapse_config'] != 'none':
            # implement both synaptic short term facilitation and depression

            self.syn_x += (self.alpha_std*(1-self.syn_x) -
                           par['dt_sec'] * self.syn_u*self.syn_x*self.h)*self.dynamic_synapse
            self.syn_u += (self.alpha_stf*(self.u-self.syn_u) +
                           par['dt_sec'] * self.u * (1-self.syn_u)*self.h)*self.dynamic_synapse
            self.syn_x.value = bm.minimum(1., bm.relu(self.syn_x))
            self.syn_u.value = bm.minimum(1., bm.relu(self.syn_u))
            h_post = self.syn_u*self.syn_x*self.h
        else:
            # no synaptic plasticity
            h_post = self.h

        self.h.value = bm.relu((1-self.alpha) * self.h) \
            + par['alpha_neuron'] * (input @ self.w_in +
                                     bm.relu(h_post) @ self.w_rnn + self.b_rnn) \
            + bm.random.normal(0, par['noise_rnn'], self.h.shape)
        self.y.value = self.h @ bm.relu(self.w_out) + self.b_out

    def predict(self, inputs):
        self.h[:] = self.init_h
        scan = bm.make_loop(body_fun=self.update,
                            dyn_vars=[self.syn_x, self.syn_u, self.h, self.y],
                            out_vars=[self.y, self.h])
        logits, hist_h = scan(inputs)
        # TODO: softmax y?
        self.y_hist[:] = logits
        return logits, hist_h

    def loss_func(self, inputs, targets, mask):
        logits, hist_h = self.predict(inputs)
        # Calculate the performance loss
        perf_loss = bp.losses.cross_entropy_loss(
            logits, targets, reduction='none') * mask
        self.perf_loss[:] = bm.mean(perf_loss)

        # L1/L2 penalty term on hidden state activity to encourage low spike rate solutions
        n = 2 if par['spike_regularization'] == 'L2' else 1
        self.spike_loss[:] = bm.mean(hist_h ** n)
        self.weight_loss[:] = bm.mean(bm.relu(self.w_rr) ** n)

        # final loss
        self.loss[:] = self.perf_loss + par['spike_cost'] * \
            self.spike_loss + par['weight_cost'] * self.weight_loss
        return self.loss.mean()
