import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from util.util import train_val_test_split, get_model, inverse_y_values, calculate_metrics, generate_cyclical_features, \
    loss_fn, remove_outlier
from torch.utils.data import DataLoader
import torch.optim as optim
from optimization.optimization import Optimization
from sequential_dataset import SequenceDataset
from matplotlib import pyplot as plt
import yaml


def data_preprocessing(config):
    df = pd.read_csv(config['data_config']['inputdata_path'])
    df.set_index("Label", inplace=True)

    # cyclical for hours using sin, cos
    df = generate_cyclical_features(df, 'Hour', 24, 0)
    print(df.head())
    print(df.info())

    # handling outlier or inf
    low_kWh, high_kWh = remove_outlier(df['Total_Energy_kWh'])
    df['Total_Energy_kWh'] = np.where(df["Total_Energy_kWh"]>high_kWh, high_kWh, df["Total_Energy_kWh"])
    df['Total_Energy_kWh'] = np.where(df["Total_Energy_kWh"]<low_kWh, low_kWh, df["Total_Energy_kWh"])
    print(df.head())

    # Data cleaning - handling missing value
    # check total number of null for each columns
    print("check if column contains null")
    print(df.isnull().sum())

    # check which column has inf or -inf
    col_name = df.columns.to_series()[np.isinf(df).any()]
    print(f"check which column contains inf: {col_name}")
    # replace infinite value
    df = df.replace([np.inf, -np.inf], 0)
    print(df.head())
    print(df.info())
    # data using l2 normalization
    # scalar_l2_norm = np.sqrt(sum(df["Total_Energy_kWh"] ** 2))
    # df["Total_Energy_kWh"] = df["Total_Energy_kWh"] / scalar_l2_norm

    df.to_csv(config['data_config']['outputdata_path'])

    return df


def train_val_test(df, ratio, config):
    # separating train, test and evaluation set
    # make sure target is being split into hours
    # X : year    month    day    hour    holiday  weekday
    # y : power consumption
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, 'Total_Energy_kWh', ratio)

    # data normalization
    scalar = MinMaxScaler()
    # the scalar of X is using training data only
    X_train_scale = scalar.fit_transform(X_train)
    X_val_scale = scalar.transform(X_val)
    X_test_scale = scalar.transform(X_test)
    # the scalar of y is using training data only
    y_train_scale = scalar.fit_transform(y_train)
    y_val_scale = scalar.transform(y_val)
    y_test_scale = scalar.transform(y_test)

    # turn data into sequential data
    sequence_length = 24

    train_dataset = SequenceDataset(
        target=y_train_scale,
        features=X_train_scale,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        target=y_test_scale,
        features=X_test_scale,
        sequence_length=sequence_length
    )
    val_dataset = SequenceDataset(
        target=y_val_scale,
        features=X_val_scale,
        sequence_length=sequence_length
    )
    # loads the data and splits them into batches for mini-batch training using Pytorch with iterable-style datasets
    train_data_loader = DataLoader(train_dataset, batch_size=config['model_config']['batch_size'], shuffle=False, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=config['model_config']['batch_size'], shuffle=False, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config['model_config']['batch_size'], shuffle=False, drop_last=True)
    return train_data_loader, test_data_loader, val_data_loader, X_test, scalar


def init_model(config):
    # init the model with parameters and move it to gpu
    model_params = {'input_dim': config['model_config']['input_dim'],
                    'hidden_dim': config['model_config']['hidden_dim'],
                    'layer_dim': config['model_config']['layer_dim'],
                    'output_dim': config['model_config']['output_dim'],
                    'dropout_prob': config['model_config']['dropout']}

    model = get_model(config['model_config']['model'], model_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def train_val_model(config, model, train_data_loader, test_data_loader, val_data_loader, X_test, scalar):
    loss = config['model_config']['loss']
    lr = config['model_config']['lr']
    weight_decay = config['model_config']['weight_decay']
    batch_size = config['model_config']['batch_size']
    epoch = config['model_config']['epoch']
    input_dim = config['model_config']['input_dim']
    output_path = config['data_config']['output_predict_path']

    # loss function being used in our model
    criterion = loss_fn(loss)

    # optimizer being used in our model
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # optimization start
    opt = Optimization(model=model, loss=criterion, optimizer=optimizer)
    opt.train(train_data_loader, val_data_loader, batch_size, epoch, input_dim)
    # plot out the losses with matplotlib
    opt.plot_loss()

    # evaluating our model
    predictions, values = opt.evaluate(test_data_loader, batch_size, input_dim)

    # inverse the predictions back to our original value
    result = inverse_y_values(predictions, values, X_test, scalar)
    result.to_csv(output_path, index=False)

    return result


def plot_result(result):
    plt.plot(result["y_test"], label="Real consumption")
    plt.plot(result["y_hat"], label="Predicted consumption")
    plt.title("y_test v.s. y_hat")
    plt.savefig('output/result_lstm_prediction.png')
    # plt.show()


def show_error_metrics(result):
    # calculate error metrics
    result_error = calculate_metrics(result)
    print(f"Result error metrics={result_error}")


if __name__ == '__main__':
    stream = open("script/config.yaml", 'r')
    cfg = yaml.load(stream)

    df = data_preprocessing(cfg)
    train_data_loader, test_data_loader, val_data_loader, X_test, scalar = train_val_test(df, cfg['data_config']['ratio'], cfg)
    model = init_model(cfg)
    result = train_val_model(cfg, model, train_data_loader, test_data_loader, val_data_loader, X_test, scalar)
    plot_result(result)
    show_error_metrics(result)

# input_dim = len(X_train.columns)  # number of features
# weight_decay = 1e-6  # l2 loss = 1e-6 , l1 loss = 0
