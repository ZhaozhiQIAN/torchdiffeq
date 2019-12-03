import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to output space
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, y0, t):
        batch_size = y0.shape[0]
        dim_y = y0.shape[-1]
        t_max = t.shape[0]
        y0 = y0.view((1, batch_size, dim_y))

        # initialize lstm with y0
        out, hidden = self.lstm(y0)
        out_linear = self.lin(out)

        out_list = [out_linear]
        # run the remaining iterations
        for t in range(t_max - 1):
            out, hidden = self.lstm(out_linear, hidden)
            out_linear = self.lin(out)
            out_list.append(out_linear)

        return torch.cat(out_list, dim=0)


class BaselineTimeLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineTimeLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim + 1, hidden_dim)

        # The linear layer that maps from hidden state space to output space
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, y0, t):
        batch_size = y0.shape[0]
        dim_y = y0.shape[-1]
        t_max = t.shape[0]
        t = t.view((t_max, batch_size, 1))

        y0 = y0.view((1, batch_size, dim_y))
        t0 = t[0:1]

        y_in = torch.cat((y0, t0), dim=-1)

        # initialize lstm with y0
        out, hidden = self.lstm(y_in)
        out_linear = self.lin(out)
        out_with_t = torch.cat((out_linear, t[1:2]), dim=-1)

        out_list = [out_linear]
        # run the remaining iterations
        for i in range(t_max - 1):
            out, hidden = self.lstm(out_with_t, hidden)
            out_linear = self.lin(out)
            if (i + 2) < t_max:
                out_with_t = torch.cat((out_linear, t[i + 2:i + 3]), dim=-1)
            out_list.append(out_linear)

        return torch.cat(out_list, dim=0)
