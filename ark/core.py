import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

from datetime import datetime

learning_rate = 0.003
num_epochs = 2
missing_rate = 0.2


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IterImputeNet(nn.Module):
  def __init__(self, num_layers=2, hidden_size=50, dropout=0.3, window_size=11):
    super(IterImputeNet, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.window_size = window_size
    self.num_cas_layers = 3  # Number of cascade layers

    # Initialize LSTMs
    self.input_size = window_size // 2 + 1
    self.forward_lstm_lst = [nn.LSTM(1, hidden_size, num_layers, batch_first=True) for _ in range(self.num_cas_layers)]
    self.backward_lstm_lst = [nn.LSTM(1, hidden_size, num_layers, batch_first=True) for _ in range(self.num_cas_layers)]
    # self.lstm = nn.LSTM(num_cols, hidden_size, num_layers, dropout=dropout, bidirectional=True)

    for i in range(self.num_cas_layers):
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

  def forward(self, forward_anchor, backward_anchor):
    for i in range(self.num_cas_layers):
      # out: (batch_size, seq_length, hidden_size*2)
      forward_lstm = self.forward_lstm_lst[i]
      backward_lstm = self.backward_lstm_lst[i]
      forward_out, _ = forward_lstm(forward_anchor)
      backward_out, _ = backward_lstm(backward_anchor)

      out = torch.cat([forward_out[:, -1,:], backward_out[:, -1,:]], dim=1).squeeze()  # (B, 100)
      out = self.dropout(out)
      out = self.fc(out)

      # Insert imputed value
      forward_anchor = forward_anchor.clone()
      backward_anchor = backward_anchor.clone()
      forward_anchor[:, -1, :] = out
      backward_anchor[:, 0, :] = out
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


class DataLoader(object):
  def __init__(self, x, window_size=11, bs=64):
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
    x = torch.from_numpy(x[:, 1]).float().to(device) # ([N])

    self.x = x
    self.y = torch.Tensor(gt_val).to(device)
    self.idx_lst = missing_idx_lst
    self.window_size = window_size
    self.bs = bs


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
        forward_anchor = self.x.index_select(0, forward_idx_lst[i:].view(-1)).to(device)
        backward_anchor = self.x.index_select(0, backward_idx_lst[i:].view(-1)).to(device)
        forward_anchor = forward_anchor.view(num_rows - i, -1, 1)
        backward_anchor = backward_anchor.view(num_rows - i, -1, 1)
        yield forward_anchor, backward_anchor, self.y[i:], idx_lst[i:]
      else:
        forward_anchor = self.x.index_select(0, forward_idx_lst[i:i+self.bs].view(-1)).to(device)
        backward_anchor = self.x.index_select(0, backward_idx_lst[i:i + self.bs].view(-1)).to(device)
        forward_anchor = forward_anchor.view(self.bs, -1, 1)  # batch size = 1 (B, N, 1)
        backward_anchor = backward_anchor.view(self.bs, -1, 1)  # batch size = 1 (B, N, 1)

      # for i, missing_idx in enumerate(self.idx_lst):
      #   forward_anchor = self.x[missing_idx - offset:missing_idx + 1].to(device) # (input_size, 1, 1)
      #   backward_anchor = self.x[missing_idx:missing_idx + offset + 1].to(device)

        yield forward_anchor, backward_anchor, self.y[i:i+self.bs], idx_lst[i:i+self.bs]

  def __setitem__(self, idx, val):
    self.x[idx] = val

  def __len__(self):
    return int(round(len(self.idx_lst) / self.bs))


df = pd.read_csv('datasets/Meteorite_Landings.csv')
cols = df.columns

# Preprocessing
# Remove name and Geolocation
df = df.drop(['name', 'GeoLocation'], axis=1)
# Process year (retrieve year)
df['year'] = df['year'].apply(lambda x: int(x.split(" ")[0].split('/')[-1]) if type(x) == str else x)
df['year'] = df['year'].astype('float')
# Categorize
df['nametype'] = df['nametype'].astype('category')
df['recclass'] = df['recclass'].astype('category')
df['fall'] = df['fall'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

# Remove rows with all NaNs
df = df.dropna(how='all')

# Df with no missing entries
df_without_null = df.dropna()
# Rows with missing entries
df_null = df[df.isnull().any(1)]

# Split training and test data
train_df, test_df = train_test_split(df_without_null, test_size=0.2)
test_df = test_df.append(df_null)

# Split training and validation
train_df, val_df = train_test_split(train_df, test_size=0.1)

# # Find missing data
# train_df_null = train_df[train_df.isna().any(axis=1)]
# percent_missing = int(round((train_df_null.shape[0] / train_df.shape[0]) * 100))

# Sort by year
train_df = train_df.sort_values('year')
year_df = train_df['year']
mass_df = train_df['mass (g)']
lat_df = train_df['reclat']
long_df = train_df['reclong']

# print((df.columns[df.isnull().any()]))
# missing_idx_lst = sorted_mass.isnull().to_numpy().nonzero()[0]

# Impute missing mass
model = IterImputeNet()
model = model.to(device)

# Loss function
loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare data
data_loader = DataLoader(train_df[['year', 'mass (g)']].values)

for epoch in range(num_epochs):
  i = 0
  for forward_anchor, backward_anchor, y, idx_lst in data_loader:
    # Forward pass
    out = model(forward_anchor, backward_anchor)

    # Calculate loss
    loss = loss_fn(out, y)

    # Backward pass and optimize
    # zero out gradient between epochs
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # Update parameters

    if (i+1) % 10 == 0:
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, len(data_loader), loss.item()))
    i += 1



