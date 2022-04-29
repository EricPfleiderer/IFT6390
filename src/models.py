import torch.nn as nn
import torch
from torch.autograd import Variable
from src.hyperparams import LSTM_space


# class LSTM(nn.Module):
#
#     def __init__(self, num_layers, hidden_size):
#         super(LSTM, self).__init__()
#
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.input_size = 1
#         self.output_size = 1
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         # Layers
#         self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
#                             num_layers=self.num_layers, batch_first=True)
#
#         for layer in self.lstm.all_weights:
#             for weight in layer:
#                 if weight.shape == 2:
#                     torch.nn.init.xavier_uniform(weight)
#
#         self.linear = nn.Linear(self.hidden_size, self.hidden_size)
#         self.linear_out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, x):
#
#         """
#         :param x: (N, L, input_size) tensor, where N is the batch size, L is the sequence length and input_size is the
#         expected number of features o f x.
#         :return:
#         """
#
#         # # (D * num_layers, batch_size, hidden_size)
#         # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))
#         #
#         # # (D * num_layers, batch_size, hidden_size)
#         # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))
#
#         # Propagate input through LSTM
#         out, (h_out, c_out) = self.lstm(x)  # , (h_0, c_0)
#
#         # Select last hidden features h_t for predictions
#         # (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
#         out = out[:, -1, :]
#
#         out = self.linear(out)
#         out = torch.relu(out)
#         out = self.linear_out(out)
#
#         return out


class LSTM(nn.Module):
    def __init__(self, num_layers, hidden_size, forecast_window):
        super(LSTM, self).__init__()
        self.input_size = 1
        self.output_size = forecast_window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm = nn.LSTM(input_size=1,hidden_size=self.hidden_size,num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc1(torch.relu(output))
        return output


def get_model_by_name(name='lstm'):

    if name.lower() == 'lstm':
        return LSTM(**LSTM_space['model'])
