import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import yaml
from util.util import train_val_test_split, get_model, inverse_y_values, calculate_metrics, generate_cyclical_features, \
    loss_fn, remove_outlier
from sequential_dataset import SequenceDataset
from optimization.optimization import Optimization
from sklearn.metrics import mean_squared_error
import optuna


# %% Data preprocessing
def data_preprocessing(config):
    df = pd.read_csv(config['data_config']['inputdata_path'])
    df.set_index("Label", inplace=True)

    # cyclical for hours using sin, cos
    df = generate_cyclical_features(df, 'Hour', 24, 0)
    # print(df.head())
    # print(df.info())

    # handling outlier or inf
    # low_kWh, high_kWh = remove_outlier(df['Total_Energy_kWh'])
    # df['Total_Energy_kWh'] = np.where(df["Total_Energy_kWh"]>high_kWh, high_kWh, df["Total_Energy_kWh"])
    # df['Total_Energy_kWh'] = np.where(df["Total_Energy_kWh"]<low_kWh, low_kWh, df["Total_Energy_kWh"])
    # print(df.head())

    # Data cleaning - handling missing value
    # check total number of null for each columns
    # print("check if column contains null")
    # print(df.isnull().sum())

    # check which column has inf or -inf
    # col_name = df.columns.to_series()[np.isinf(df).any()]
    # print(f"check which column contains inf: {col_name}")
    # replace infinite value
    # df = df.replace([np.inf, -np.inf], 0)
    # print(df.head())
    # print(df.info())
    # data using l2 normalization
    # scalar_l2_norm = np.sqrt(sum(df["Total_Energy_kWh"] ** 2))
    # df["Total_Energy_kWh"] = df["Total_Energy_kWh"] / scalar_l2_norm

    # df.to_csv(config['data_config']['outputdata_path'])

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
    train_data_loader = DataLoader(train_dataset, batch_size=config['model_config']['batch_size'], shuffle=True,
                                   drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=config['model_config']['batch_size'], shuffle=False,
                                  drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config['model_config']['batch_size'], shuffle=True,
                                 drop_last=True)
    return train_data_loader, test_data_loader, val_data_loader, X_test, scalar


# %% Build a  Model
def init_model(config, params):
    # init the model with parameters and move it to gpu
    # input_dim = num of features
    model_params = {'input_dim': config['model_config']['input_dim'],
                    'hidden_dim': params["n_hidden"],
                    'layer_dim': params["n_layers"],
                    'output_dim': config['model_config']['output_dim'],
                    'dropout_prob': params["dropout"]}

    model = get_model(config['model_config']['model'], model_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


# %% Training

def train_val_model(config, params, model, train_data_loader, test_data_loader, val_data_loader, X_test, scalar):
    loss = config['model_config']['loss']
    lr = params['lr']
    weight_decay = config['model_config']['weight_decay']
    batch_size = config['model_config']['batch_size']
    epoch = config['model_config']['epoch']
    input_dim = config['model_config']['input_dim']

    # loss function being used in our model
    criterion = loss_fn(loss)

    # optimizer being used in our model
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # optimization start
    opt = Optimization(model=model, loss=criterion, optimizer=optimizer)
    best_epoch = opt.train(train_data_loader, val_data_loader, epoch)
    # plot out the losses with matplotlib
    opt.plot_loss()

    # testing our model
    predictions, values = opt.test(test_data_loader,best_epoch=best_epoch)
    # inverse the predictions back to our original value
    result = inverse_y_values(predictions, values, X_test, scalar)

    return result


def plot_result(result):
    plt.plot(result["y_test"][-100:], 'r', label="Real consumption")
    plt.plot(result["y_hat"][-100:], 'g', label="Predicted consumption")
    plt.title("y_test v.s. y_hat")
    plt.legend()
    plt.savefig('output/result_lstm_prediction.png')
    # plt.show()


def show_error_metrics(result):
    # calculate error metrics
    result_error = calculate_metrics(result)
    print(f"Result error metrics={result_error}")


def objective(trial):
    params = {
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "n_hidden": trial.suggest_int("n_hidden", 16, 64),
        "dropout": trial.suggest_uniform("dropout", 0.3, 0.8),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2)
    }
    return run_all(config, params)


def run_all(config, params):
    df = data_preprocessing(config)
    train_data_loader, test_data_loader, val_data_loader, X_test, scalar = train_val_test(df, config['data_config'][
        'ratio'], config)
    model = init_model(config, params)
    result = train_val_model(config, params, model, train_data_loader, test_data_loader, val_data_loader, X_test,
                             scalar)
    rmse = mean_squared_error(result.y_test, result.y_hat) ** 0.5
    return rmse


def run_best_param(config, params):
    df = data_preprocessing(config)
    train_data_loader, test_data_loader, val_data_loader, X_test, scalar = train_val_test(df, config['data_config'][
        'ratio'], config)
    model = init_model(config, params)
    result = train_val_model(config, params, model, train_data_loader, test_data_loader, val_data_loader, X_test,
                             scalar)
    output_path = config['data_config']['output_predict_path']
    result.to_csv(output_path, index=False)
    plot_result(result)
    show_error_metrics(result)


if __name__ == '__main__':
    stream = open("script/config.yaml", 'r')  # all setting parameters are writen on this script
    config = yaml.load(
        stream)  # configuration files are files used to configure the parameters and initial settings for some computer programs.

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000)
    print("best trial:")
    trial_ = study.best_trial

    print(trial_.values)
    print(trial_.params)
    params = {
        "n_layers": trial_.params["n_layers"],
        "n_hidden": trial_.params["n_hidden"],
        "dropout": trial_.params["dropout"],
        "lr": trial_.params["lr"]
    }
    run_best_param(config, params)
