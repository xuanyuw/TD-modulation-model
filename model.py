
import torch
import numpy as np
from parameters import *
from model_func import *
from torch.nn.parameter import Parameter


class Model(torch.nn.Module):

    def __init__(self, device, par=par, train=True):
        super(Model, self).__init__()

        self.syn_x_init = torch.tensor(par['syn_x_init'], requires_grad=False)
        self.syn_u_init = torch.tensor(par['syn_u_init'], requires_grad=False)
        self.x_init = torch.tensor(par['x0'], requires_grad=True)
        self.device = device
        self.train_mode = train
        self.x_grad = []

        if not train:
            state_dict = torch.load(par['save_dir'] + par['weight_fn'])
            for k, v in state_dict.items():
                self[k] = v

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __getitem__(self, k):
        getattr(self, k)

    def forward(self, input_data, target_data, mask, train=True):

        # Load the input activity, the target data, and the training mask for this batch of trials
        input_data = torch.unbind(input_data, axis=0)
        self.target_data = target_data
        self.mask = mask
        # Build the TensorFlow graph
        self.run_model(input_data, train=train)
        return self.y_output, self.x

    def run_model(self, input_data, train):
        self.x = []
        self.r = []
        self.syn_x = []
        self.syn_u = []
        self.y = []

        x = self.x_init.to(self.device)
        syn_x = self.syn_x_init.to(self.device)
        syn_u = self.syn_u_init.to(self.device)
        relu = torch.nn.ReLU(inplace=False)
        r = relu(x).to(self.device)

        w_in, w_rnn = self.update_weights(train)

        # Loop through the neural inputs to the RNN, indexed in time
        for rnn_input in input_data:
            self.rnn_input = rnn_input
            x, r, syn_x, syn_u = self.rnn_cell(
                rnn_input, x, r, syn_x, syn_u, w_in, w_rnn)

            self.x.append(x)
            self.r.append(r)
            self.syn_x.append(syn_x)
            self.syn_u.append(syn_u)
            w_out = torch.multiply(self.w_out, torch.tensor(
                par['w_out_mask']).to(self.device)).float()
            self.y.append(
                r @ relu(w_out) + self.b_out)

        # Stack outputs and return
        self.x = torch.stack(self.x)
        # reshape to make the batch size the first dimension
        self.x = self.x.permute(1, 0, 2)
        if par['regularization'] or self.train_mode:
            self.x.retain_grad()
        self.r = torch.stack(self.r)
        # reshape to make the batch size the first dimension
        self.r = self.r.permute(1, 0, 2)
        with torch.no_grad():
            self.syn_x = torch.stack(self.syn_x)
            self.syn_x = self.syn_x.permute(1, 0, 2)
            self.syn_u = torch.stack(self.syn_u)
            self.syn_u = self.syn_u.permute(1, 0, 2)
        self.y = torch.stack(self.y)
        self.y = self.y.permute(1, 0, 2)
        softmax = torch.nn.Softmax(dim=0)
        self.y_output = softmax(self.y)
        self.y_output

    def rnn_cell(self, rnn_input, x, r, syn_x, syn_u, w_in, w_rnn):
        # Update neural activity and short-term synaptic plasticity values
        # Update the synaptic plasticity paramaters
        inplace_relu = torch.nn.ReLU(inplace=True)
        relu = torch.nn.ReLU(inplace=False)
        if par['synapse_config'] != 'none':
            # implement both synaptic short term facilitation and depression

            with torch.no_grad():
                dynamic_synapse = torch.tensor(
                    par['dynamic_synapse']).to(self.device)
                alpha_std = torch.tensor(par['alpha_std']).to(self.device)
                alpha_stf = torch.tensor(par['alpha_stf']).to(self.device)
                U = torch.tensor(par['U']).to(self.device)

                syn_x = syn_x + (alpha_std*(1-syn_x) -
                                 par['dt_sec'] * syn_u*syn_x*r)*dynamic_synapse
                syn_u = syn_u + (alpha_stf*(U-syn_u) +
                                 par['dt_sec'] * U*(1-syn_u)*r)*dynamic_synapse
                syn_x = torch.min(torch.tensor(np.array([1])).float().to(
                    self.device), inplace_relu(syn_x))
                syn_u = torch.min(torch.tensor(np.array([1])).float().to(
                    self.device), inplace_relu(syn_u))

                del dynamic_synapse, alpha_std, alpha_stf, U

            x_post = syn_u*syn_x*x
        else:
            # no synaptic plasticity
            x_post = x

        x = (1-par['alpha_neuron']) * x_post \
            + par['alpha_neuron'] * (rnn_input @ w_in + relu(x_post) @ w_rnn + self.b_rnn) \
            + torch.normal(0, par['noise_rnn'], size=x.shape).to(self.device)
        r = inplace_relu(x)
        return x, r, syn_x, syn_u

    def load_weights(self, par=par):
        # load all weights. biases, and initial values

        var_dict = {k[:-1]: par[k]
                    for k in par.keys() if (k[-1] == '0') & (k != 'x0')}
        # all keys in par with a suffix of '0' are initial values of trainable variables
        for k, v in var_dict.items():
            if k == 'w_in':
                self[k] = torch.tensor(v, requires_grad=False)
            else:
                self[k] = Parameter(torch.tensor(v), requires_grad=True)
        # ensure excitatory neurons only have postive outgoing weights,
        # and inhibitory neurons have negative outgoing weights
        # send to gpu
        self.w_rnn_fix_t1r = torch.zeros_like(self.w_rnn_base).to(self.device)
        self.w_rnn_fix_t2r = torch.zeros_like(self.w_rnn_base).to(self.device)
        self.w_rnn_base = Parameter(self.w_rnn_base.to(self.device))
        self.b_rnn = Parameter(self.b_rnn.to(self.device))
        self.w_out = Parameter(self.w_out.to(self.device))
        self.b_out = Parameter(self.b_out.to(self.device))
        self.w_in = self.w_in.to(self.device)

    def update_weights(self, train):
        relu = torch.nn.ReLU(inplace=False)
        # if not train:
        #     par['w_out_mask'] = par['w_out_mask0']
        #     w_in = torch.multiply(self.w_in, torch.tensor(
        #         par['w_in_mask0']).to(self.device)).float()
        #     w_rnn = torch.multiply(self.w_rnn_base, torch.tensor(
        #         par['conn_mask0']).to(self.device)).float()
        #     w_rnn = torch.matmul(torch.tensor(par['EI_matrix']).to(
        #         self.device).float(), relu(w_rnn).float())
        # else:
        if self.cond == "t1r":
            w_in = torch.multiply(self.w_in, torch.tensor(
                par['w_in_mask_t1r']).to(self.device)).float()
            # w_rnn = w_rnn_plastic + w_rnn_fixed, w_rnn_plastic can be updated
            if train:
                w_rnn_plastic = torch.multiply(self.w_rnn_base, torch.tensor(
                    par['fix_mask_t2r']).to(self.device)).float()
                self.w_rnn_base = Parameter(torch.multiply(torch.tensor(
                    par['conn_mask_t1r']).to(self.device), w_rnn_plastic + self.w_rnn_fix_t2r))
            par['w_out_mask'] = par['w_out_mask_t1r']
        else:
            w_in = torch.multiply(self.w_in, torch.tensor(
                par['w_in_mask_t2r']).to(self.device)).float()
            # w_rnn = w_rnn_plastic + w_rnn_fixed, w_rnn_plastic can be updated
            if train:
                w_rnn_plastic = torch.multiply(self.w_rnn_base, torch.tensor(
                    par['fix_mask_t1r']).to(self.device)).float()
                self.w_rnn_base = Parameter(torch.multiply(torch.tensor(
                    par['conn_mask_t2r']).to(self.device), w_rnn_plastic + self.w_rnn_fix_t1r))
            par['w_out_mask'] = par['w_out_mask_t2r']
        w_rnn = torch.matmul(torch.tensor(par['EI_matrix']).to(
            self.device).float(), relu(self.w_rnn_base).float())
        return w_in, w_rnn
