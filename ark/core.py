import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split

from datetime import datetime
from impute_net import IterImputeNet
from data_loader import DataLoader

learning_rate = 0.003
num_epochs = 20
missing_rate = 0.2
num_cas_layers = 3
batch_size = 32


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
train_df = df_without_null
test_df = df_null
# train_df, test_df = train_test_split(df_without_null, test_size=0.2)
# test_df = test_df.append(df_null)

# Split training and validation
# train_df, val_df = train_test_split(train_df, test_size=0.1)

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
model = IterImputeNet(num_cas_layers, device=device)
model = model.to(device)

# Loss function
loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare data
data_loader = DataLoader(train_df[['year', 'mass (g)']].values, missing_rate, bs=batch_size, device=device)

for layer_num in range(num_cas_layers):
  # Train model
  for epoch in range(num_epochs):
    i = 0
    for forward_anchor, backward_anchor, y, _ in data_loader:
      # Forward pass
      out = model(forward_anchor, backward_anchor, layer_num)

      # Calculate loss
      loss = loss_fn(out, y)

      # Backward pass and optimize
      # zero out gradient between epochs
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()  # Update parameters

      if (i+1) % 50 == 0:
        print ('Layer [{}/{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(layer_num + 1, num_cas_layers, epoch + 1, num_epochs, i + 1, len(data_loader), loss.item()))
      i += 1

  # Predict
  with torch.no_grad():
    for forward_anchor, backward_anchor, y, idx_lst in data_loader:
      out = model(forward_anchor, backward_anchor, layer_num)
      data_loader[idx_lst] = out



