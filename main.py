import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from util import train_val_test_split, get_model, format_predictions, calculate_metrics
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from optimization import Optimization
from sequential_dataset import SequenceDataset

df = pd.read_csv('data.csv')

torch.manual_seed(99)

# separating train, test and evaluation set
# make sure target is being split into hours
# X : year    month    day    hour    holiday
# y : power consumption
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, 'power_consumption', 0.2)

# data normalization
scaler = MinMaxScaler()
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)
y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)

# turn data into sequential data
# TODO: sequence length should be discussed
sequence_length = 30

train_dataset = SequenceDataset(
    target=X_train_arr,
    features=y_train_arr,
    sequence_length=sequence_length
)
test_dataset = SequenceDataset(
    target=X_test_arr,
    features=y_test_arr,
    sequence_length=sequence_length
)
val_dataset = SequenceDataset(
    target=X_val_arr,
    features=y_val_arr,
    sequence_length=sequence_length
)

# loads the data and splits them into batches for mini-batch training using Pytorch with iterable-style datasets
batch_size = 512

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

input_dim = len(X_train.columns)
output_dim = 1
hidden_dim = 16
layer_dim = 2
batch_size = 512
dropout = 0.3
n_epochs = 30
lr = 1e-3
weight_decay = 1e-6

model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'dropout_prob': dropout}

model = get_model('lstm', model_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss function being used in our model
criterion = nn.MSELoss(reduction="mean")

# optimizer being used in our model
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = optim.SGD(lstm.parameters(), lr=learning_rate)

# optimization start
opt = Optimization(model=model, loss_fn=criterion, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)

# plot out the losses with matplotlib
opt.plot_losses()

# testing our model
predictions, values = opt.evaluate(test_loader, batch_size=batch_size, n_features=input_dim)

# inverse the predictions back to our original value
df_result = format_predictions(predictions, values, X_test, scaler)
# df_result.to_csv(r'data\export_result.csv', index=False)

# calculate error metrics
result_metrics = calculate_metrics(df_result)
