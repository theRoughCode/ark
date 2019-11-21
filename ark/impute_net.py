import torch
import torch.nn as nn

class IterImputeNet(nn.Module):
  def __init__(self, num_features=1, hidden_size=50, dropout=0.3, window_size=11, device='cuda'):
    super(IterImputeNet, self).__init__()
    self.hidden_size = hidden_size
    self.window_size = window_size
    self.device = device

    # Initialize LSTMs
    self.input_size = window_size // 2 + 1
    self.forward_lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
    self.backward_lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
    # self.lstm = nn.LSTM(num_cols, hidden_size, dropout=dropout, bidirectional=True)

    # Initialize weights
    nn.init.xavier_uniform_(self.forward_lstm.weight_ih_l0)
    nn.init.orthogonal_(self.forward_lstm.weight_hh_l0)
    nn.init.xavier_uniform_(self.backward_lstm.weight_ih_l0)
    nn.init.orthogonal_(self.backward_lstm.weight_hh_l0)
    # Initialize bias
    nn.init.constant_(self.forward_lstm.bias_ih_l0, 0)
    nn.init.constant_(self.forward_lstm.bias_hh_l0, 0)
    nn.init.constant_(self.backward_lstm.bias_ih_l0, 0)
    nn.init.constant_(self.backward_lstm.bias_hh_l0, 0)

    self.dropout = nn.Dropout(p=dropout)
    self.fc = nn.Linear(hidden_size * 2, 1)
    self.activation = nn.LeakyReLU()

  def forward(self, forward_anchor, backward_anchor):
    # out: (batch_size, seq_length, hidden_size*2)
    forward_out, _ = self.forward_lstm(forward_anchor)
    backward_out, _ = self.backward_lstm(backward_anchor)

    out = torch.cat([forward_out[:, -1, :], backward_out[:, -1, :]], dim=1).squeeze()  # (B, 100)
    out = self.dropout(out)
    out = self.fc(out)
    out = self.activation(out)

    return out.squeeze()