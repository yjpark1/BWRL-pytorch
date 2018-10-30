import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x, h0=None):
        if len(x.size()) == 3:
            # TimeDistributed + Dense
            # shape: (seq_len, batch, features)
            seq_len, b, n = x.size(0), x.size(1), x.size(2)
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(seq_len * b, n)

        elif len(x.size()) == 4:
            # TimeDistributed + RNN
            # shape: (seq_len, batch, agents, features)
            seq_len, b, n, d = x.size(0), x.size(1), x.size(2), x.size(3)
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(seq_len, b * n, d)

            # return
        if isinstance(self.module, nn.Linear):
            # TimeDistributed + Dense
            y = self.module(x_reshape)
            # We have to reshape Y
            # shape: (seq_len, batch, documents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            return y

        elif isinstance(self.module, nn.LSTM):
            # TimeDistributed + RNN
            if h0 is not None:
                h0, c0 = h0
                h0 = h0.contiguous().view(h0.size()[0], b * n, h0.size()[-1])
                c0 = c0.contiguous().view(c0.size()[0], b * n, c0.size()[-1])
                h0 = (h0, c0)

            y, (h, c) = self.module(x_reshape, h0)
            # shape: (seq_len, batch x agents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            # shape: (seq_len, batch, agents, features)

            # shape: (batch, agents, features)
            h = h.contiguous().view(h.size()[0], b, n, h.size()[-1])
            c = c.contiguous().view(c.size()[0], b, n, c.size()[-1])
            return y, (h, c)

        elif isinstance(self.module, nn.GRU):
            # TimeDistributed + RNN
            y, h = self.module(x_reshape, h0)
            # shape: (seq_len, batch x agents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            # shape: (seq_len, batch, agents, features)

            # shape: (batch, agents, features)
            h = h.contiguous().view(1, b, n, h.size()[-1])
            return y, h

        else:
            raise ImportError('Not Supported Layers!')


class ActorNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, nb_agents, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int) : Number of dimensions in input  (agents, observation)
            out_dim (int)   : Number of dimensions in output
            hidden_dim (int) : Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ActorNetwork, self).__init__()
        self.nb_agents = nb_agents
        self.dense1 = nn.Linear(input_dim, 64)
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.bilstm = nn.LSTM(64, 32, num_layers=1, bidirectional=True)
        self.dense2_cont = nn.Linear(64, out_dim[0])
        self.dense2_disc = nn.Linear(64, out_dim[1])
        self.dense3 = nn.Linear(64, input_dim)

    def forward(self, obs):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): policy, next_state
        """
        hid = F.relu(self.dense1(obs))
        hid, _ = self.bilstm(hid, None)
        hid = F.relu(hid)
        policy_cont = self.dense2_cont(hid)
        policy_cont = F.tanh(policy_cont)
        policy_disc = self.dense2_disc(hid)
        policy_disc = nn.Softmax(dim=-1)(policy_disc)
        next_state = self.dense3(hid)
        return [policy_cont, policy_disc], next_state


class CriticNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, nb_agents, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int): Number of dimensions in input  (agents, observation)
            out_dim (int): Number of dimensions in output

            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(CriticNetwork, self).__init__()
        self.nb_agents = nb_agents
        self.dense1 = nn.Linear(input_dim, 64)
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.lstm = nn.LSTM(64, 64, num_layers=1, bidirectional=False)
        self.dense2 = nn.Linear(64, out_dim)
        self.dense3 = nn.Linear(64, out_dim)

    def forward(self, obs, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Q-function
            out (PyTorch Matrix): reward
        """
        obs_act = torch.cat((obs, action), dim=-1)
        hid = F.relu(self.dense1(obs_act))
        hid, _ = self.lstm(hid, None)
        hid = F.relu(hid[:, -1, :])
        Q = self.dense2(hid)
        r = self.dense3(hid)
        return Q, r
