import torch
import numpy as np
from sklearn.impute import IterativeImputer

class DataLoader(object):
  def __init__(self, x, missing_rate, window_size=11, bs=64, device='cuda'):
    # Get ground truth
    num_gt = int(round(missing_rate * x.shape[0]))
    missing_idx_lst = np.random.choice(x.shape[0], num_gt)

    # Remove edge cases
    num_rows = len(x)
    min_index = window_size // 2
    max_index = num_rows - min_index - 1
    missing_idx_lst = missing_idx_lst[missing_idx_lst >= min_index]
    missing_idx_lst = missing_idx_lst[missing_idx_lst <= max_index]
    missing_idx_lst = torch.LongTensor(missing_idx_lst).to(device)

    # Set missing values
    gt_val = x[missing_idx_lst, 1]
    x[missing_idx_lst, 1] = np.nan
    # Get initial imputed values
    imp = IterativeImputer(max_iter=int(missing_rate * 100), random_state=0)
    x = imp.fit_transform(x)

    # Convert to tensor
    x = torch.from_numpy(x[:, 1]).float().to(device)  # ([N])

    self.x = x
    self.y = torch.Tensor(gt_val).to(device)
    self.idx_lst = missing_idx_lst
    self.window_size = window_size
    self.bs = bs
    self.device = device


  def __iter__(self):
    offset = self.window_size // 2
    num_rows = len(self.idx_lst)
    count = 0

    num_rows = self.idx_lst.size(0)
    input_size = self.window_size // 2 + 1
    # (M, input_size)
    idx_lst = self.idx_lst.unsqueeze(-1).repeat(1, input_size)
    forward_idx_lst = idx_lst + torch.arange(1 - input_size, 1).unsqueeze(0).repeat(num_rows, 1)
    backward_idx_lst = idx_lst + torch.arange(0, input_size).unsqueeze(0).repeat(num_rows, 1)

    for i in range(0, num_rows, self.bs):
      if i + self.bs > num_rows:
        forward_anchor = self.x.index_select(0, forward_idx_lst[i:].view(-1)).to(self.device)
        backward_anchor = self.x.index_select(0, backward_idx_lst[i:].view(-1)).to(self.device)
        forward_anchor = forward_anchor.view(num_rows - i, -1, 1)
        backward_anchor = backward_anchor.view(num_rows - i, -1, 1)
        yield forward_anchor, backward_anchor, self.y[i:], self.idx_lst[i:]
      else:
        forward_anchor = self.x.index_select(0, forward_idx_lst[i:i+self.bs].view(-1)).to(self.device)
        backward_anchor = self.x.index_select(0, backward_idx_lst[i:i + self.bs].view(-1)).to(self.device)
        forward_anchor = forward_anchor.view(self.bs, -1, 1)  # batch size = 1 (B, N, 1)
        backward_anchor = backward_anchor.view(self.bs, -1, 1)  # batch size = 1 (B, N, 1)

      # for i, missing_idx in enumerate(self.idx_lst):
      #   forward_anchor = self.x[missing_idx - offset:missing_idx + 1].to(self.device) # (input_size, 1, 1)
      #   backward_anchor = self.x[missing_idx:missing_idx + offset + 1].to(self.device)

        yield forward_anchor, backward_anchor, self.y[i:i+self.bs], self.idx_lst[i:i+self.bs]

  def __setitem__(self, idx, val):
    self.x[idx] = val

  def __len__(self):
    return int(round(len(self.idx_lst) / self.bs))