from init_weight import initialize_weights
import brainpy as bp
import brainpy.math as bm
from numpy import load
from numpy.random import normal
from os.path import join
bp.math.set_platform('cpu')


class Model(bp.layers.Module):

    def __init__(self, par, stim, train=True):
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
        self.EI_matrix = bm.array(par['EI_matrix'])

        # self.n_hidden = par['n_hidden']
        # self.n_total = par['n_total']

        self.y = bm.Variable(
            bm.ones((par['batch_size'], par['n_output'])))
        self.y_hist = bm.Variable(
            bm.zeros((par['num_time_steps'], par['batch_size'], par['n_output'])))
        self.h_hist = bm.Variable(bm.zeros((par['num_time_steps'], par['batch_size'], par['n_total'])))

        # Loss
        self.loss = bm.Variable(bm.zeros(1))
        self.perf_loss = bm.Variable(bm.zeros(1))
        self.spike_loss = bm.Variable(bm.zeros(1))
        self.weight_loss = bm.Variable(bm.zeros(1))

        # weights 
        if train:
            all_weights = initialize_weights(par['learning_rate'], par['rep'], stim)
        else:
            all_weights = load(
                join(par['save_dir'], par['weight_fn']), allow_pickle=True)
            all_weights = all_weights.item()
        self.in_mask = bm.array(all_weights['in_mask_init'])
        self.rnn_mask = bm.array(all_weights['rnn_mask_init'])
        self.out_mask = bm.array(all_weights['out_mask_init'])
        # self.init_w_rnn = bm.array(all_weights['w_rnn0'])
        # self.w_in = bm.TrainVar(all_weights['w_in0'])
        self.w_in = bm.Variable(all_weights['w_in0'])
        self.w_rnn = bm.TrainVar(all_weights['w_rnn0'])
        # self.w_out = bm.TrainVar(all_weights['w_out0'])
        self.w_out = bm.Variable(all_weights['w_out0'])
        self.b_rnn = bm.TrainVar(all_weights['b_rnn0'])
        self.b_out = bm.Variable(all_weights['b_out0'])

        # Constants

        self.u_init = bm.array(par['U'])
        self.syn_x_init = bm.array(par['syn_x_init'])
        self.syn_u_init = bm.array(par['syn_u_init'])
        self.dt_sec = par['dt_sec']
        self.noise_rnn = par['noise_rnn']
        self.spike_regularization = par['spike_regularization']
        self.weight_cost = par['weight_cost']
        self.spike_cost = par['spike_cost']
        self.synapse_config = par['synapse_config']

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __getitem__(self, k):
        getattr(self, k)

    def reset(self):

        self.u.value = self.u_init
        self.syn_x.value = self.syn_x_init
        self.syn_u.value = self.syn_u_init
        self.loss[:] = 0.
        self.perf_loss[:] = 0.
        self.spike_loss[:] = 0.
        self.weight_loss[:] = 0.

    def update(self, input, **kwargs):
        if self.synapse_config != 'none':
            # implement both synaptic short term facilitation and depression

            self.syn_x += (self.alpha_std*(1-self.syn_x) -
                           self.dt_sec * self.syn_u*self.syn_x*self.h)*self.dynamic_synapse
            self.syn_u += (self.alpha_stf*(self.u-self.syn_u) +
                           self.dt_sec * self.u * (1-self.syn_u)*self.h)*self.dynamic_synapse
            self.syn_x.value = bm.minimum(1., bm.relu(self.syn_x))
            self.syn_u.value = bm.minimum(1., bm.relu(self.syn_u))
            h_post = self.syn_u*self.syn_x*self.h
        else:
            # no synaptic plasticity
            h_post = self.h

        # Update the hidden state. Only use excitatory projections from input layer to RNN
        # All input and RNN activity will be non-negative
        
     
        w_rnn = bm.relu(self.w_rnn)
        # replace interneuron weights with original ones
        # w_rnn = w_rnn.at[self.n_hidden : self.n_total, : self.n_hidden].set(self.init_w_rnn[self.n_hidden : self.n_total, : self.n_hidden])
        # w_rnn = w_rnn.at[: self.n_hidden, self.n_hidden : self.n_total].set(self.init_w_rnn[: self.n_hidden, self.n_hidden : self.n_total])

        w_rnn =  self.EI_matrix @ w_rnn



        state = self.alpha * bm.relu(input @ bm.relu(self.w_in) + h_post @ w_rnn +
                              self.b_rnn) + normal(0, self.noise_rnn, self.h.shape)
        self.h.value = state + self.h * (1 - self.alpha)
        self.y.value = self.h @ bm.relu(self.w_out) + self.b_out

    def predict(self, inputs, train):
        if train:
            self.h[:] = self.init_h
        else:
            self.h = self.init_h
        scan = bm.make_loop(body_fun=self.update,
                            dyn_vars=[self.syn_x, self.syn_u, self.h, self.y],
                            out_vars=[self.y, self.h])
        logits, hist_h = scan(inputs)
        # TODO: softmax y?
        if train:
            self.y_hist[:] = logits
            self.h_hist[:] = hist_h
        # self.y_hist = logits

        return logits, hist_h

    def loss_func(self, inputs, targets, mask):
        logits, hist_h = self.predict(inputs, True)
        # Calculate the performance loss
        perf_loss = bp.losses.cross_entropy_loss(
            bm.softmax(logits), targets, reduction='none') * mask
        self.perf_loss[:] = bm.mean(perf_loss)
        # self.perf_loss = bm.mean(perf_loss)

        # L1/L2 penalty term on hidden state activity to encourage low spike rate solutions
        n = 2 if self.spike_regularization == 'L2' else 1
        self.spike_loss[:] = bm.mean(hist_h ** n)
        self.weight_loss[:] = bm.mean(bm.relu(self.w_rnn) ** n)
        # self.spike_loss = bm.mean(hist_h ** n)
        # self.weight_loss = bm.mean(bm.relu(self.w_rnn) ** n)

        # final loss
        # self.loss[:] = self.perf_loss
        self.loss[:] = self.perf_loss + self.spike_cost * \
            self.spike_loss + self.weight_cost * self.weight_loss
        # self.loss = self.perf_loss + self.spike_cost * \
        #     self.spike_loss + self.weight_cost * self.weight_loss

        return self.loss.mean()

    def get_all_masks(self):
        all_masks = {}
        all_masks['in_mask_init'] = self.in_mask
        all_masks['rnn_mask_init'] = self.rnn_mask
        all_masks['out_mask_init'] = self.out_mask
        return all_masks
