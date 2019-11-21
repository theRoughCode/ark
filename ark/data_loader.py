import torch
import torch.nn.functional as F
import numpy as np
from sklearn.impute import IterativeImputer

class DataLoader(object):
  def __init__(self, x, missing_rate=0.2, is_test=False, window_size=11, bs=64, device='cuda'):
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
    gt_val = x[missing_idx_lst, -1]
    x[missing_idx_lst, -1] = np.nan
    # Get initial imputed values
    imp = IterativeImputer(max_iter=int(missing_rate * 100), random_state=0)
    x = imp.fit_transform(x)

    # Convert to tensor
    x = torch.from_numpy(x).float().to(device)  # (N, D)

    self.x = x
    self.num_rows, self.features = x.shape
    self.y = torch.Tensor(gt_val).to(device)
    self.idx_lst = missing_idx_lst
    self.window_size = window_size
    self.bs = bs
    self.device = device


  def __iter__(self):
    num_rows = len(self.idx_lst)
    input_size = self.window_size // 2 + 1
    # (M, input_size)
    idx_lst = self.idx_lst.unsqueeze(-1).repeat(1, input_size)
    forward_idx_lst = idx_lst + torch.arange(1 - input_size, 1).unsqueeze(0).repeat(num_rows, 1)
    backward_idx_lst = idx_lst + torch.arange(input_size - 1, -1, -1).unsqueeze(0).repeat(num_rows, 1)

    for i in range(0, num_rows, self.bs):
      end_idx = i + self.bs if i + self.bs <= num_rows else num_rows
      forward_anchor = self.x.index_select(0, forward_idx_lst[i:end_idx].view(-1)).to(self.device)
      backward_anchor = self.x.index_select(0, backward_idx_lst[i:end_idx].view(-1)).to(self.device)
      # (B, seq_len, D)
      forward_anchor = forward_anchor.view(end_idx - i, -1, self.features)
      backward_anchor = backward_anchor.view(end_idx - i, -1, self.features)

      val = self.y[i:end_idx]
      if i + self.bs > num_rows:
        diff = i + self.bs - num_rows
        forward_anchor = F.pad(forward_anchor, (0, 0, 0, 0, 0, diff))
        backward_anchor = F.pad(backward_anchor, (0, 0, 0, 0, 0, diff))
        val = F.pad(val, (0, diff))

      yield forward_anchor, backward_anchor, val, self.idx_lst[i:end_idx]

  def __setitem__(self, idx, val):
    if val.shape[0] > self.x[idx].shape[0]:
      val = val[:self.x[idx].shape[0]]
    self.x[idx] = val

  def __getitem__(self, idx):
    return self.x[idx]

  def __len__(self):
    return int(round(len(self.idx_lst) / self.bs))