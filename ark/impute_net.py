import torch
import torch.nn as nn

class IterImputeNet(nn.Module):
  def __init__(self, num_layers=3, hidden_size=50, dropout=0.3, window_size=11, device='cuda'):
    super(IterImputeNet, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.window_size = window_size
    self.device = device

    # Initialize LSTMs
    self.input_size = window_size // 2 + 1
    self.forward_lstm_lst = [nn.LSTM(1, hidden_size, batch_first=True) for _ in range(self.num_layers)]
    self.backward_lstm_lst = [nn.LSTM(1, hidden_size, batch_first=True) for _ in range(self.num_layers)]
    # self.lstm = nn.LSTM(num_cols, hidden_size, dropout=dropout, bidirectional=True)

    for i in range(self.num_layers):
      # Initialize weights
      nn.init.xavier_uniform_(self.forward_lstm_lst[i].weight_ih_l0)
      nn.init.orthogonal_(self.forward_lstm_lst[i].weight_hh_l0)
      nn.init.xavier_uniform_(self.backward_lstm_lst[i].weight_ih_l0)
      nn.init.orthogonal_(self.backward_lstm_lst[i].weight_hh_l0)
      # Initialize bias
      nn.init.constant_(self.forward_lstm_lst[i].bias_ih_l0, 0)
      nn.init.constant_(self.forward_lstm_lst[i].bias_hh_l0, 0)
      nn.init.constant_(self.backward_lstm_lst[i].bias_ih_l0, 0)
      nn.init.constant_(self.backward_lstm_lst[i].bias_hh_l0, 0)

    self.dropout = nn.Dropout(p=dropout)
    self.fc = nn.Linear(hidden_size * 2, 1)
    self.softmax = nn.LogSoftmax()

  def forward(self, forward_anchor, backward_anchor, layer_num):
    forward_lstm = self.forward_lstm_lst[layer_num]
    backward_lstm = self.backward_lstm_lst[layer_num]

    # out: (batch_size, seq_length, hidden_size*2)
    forward_out, _ = forward_lstm(forward_anchor)
    backward_out, _ = backward_lstm(backward_anchor)

    out = torch.cat([forward_out[:, -1,:], backward_out[:, -1,:]], dim=1).squeeze()  # (B, 100)
    out = self.dropout(out)
    out = self.fc(out)
    out = self.softmax(out)

    return out.squeeze()

  # def __iter__(self, x, idx_lst):
  #   offset = self.window_size // 2
  #   for missing_idx in idx_lst:
  #     forward_anchor = x[missing_idx - offset:missing_idx + 1].to(device) # (input_size, 1, 1)
  #     backward_anchor = x[missing_idx:missing_idx + offset + 1].to(device)
  #     yield forward_anchor, backward_anchor
      # out = self.forward(forward_anchor, backward_anchor)
      # Insert imputed value
      # x[missing_idx] = out
    # return x

    # num_missing = idx_lst.size(0)
    # idx_lst = self.missing_idx_lst.unsqueeze(-1).repeat(1, self.input_size)
    # forward_idx_lst = idx_lst + torch.arange(1 - self.input_size, 1).unsqueeze(0).repeat(num_missing, 1)
    # backward_idx_lst = idx_lst + torch.arange(0, self.input_size).unsqueeze(0).repeat(num_missing, 1)
    # forward_anchor = self.data[forward_idx_lst]
    # backward_anchor = self.data[backward_idx_lst]

    # out = self.forward(forward_anchor, backward_anchor)