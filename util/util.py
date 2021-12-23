from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.LSTM import LSTMModel
from models.GRU import GRUModel
from models.BiLSTM import BiLSTM
import numpy as np
import pandas as pd
import torch.nn as nn


def train_val_test_split(df, target, ratio):
    X = df.drop(columns=[target])
    y = df[[target]]
    # separate the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, shuffle=False)
    # separate the validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(ratio / (1 - ratio)), shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model(model, model_params):
    models = {
        "lstm": LSTMModel,
        "bilstm": BiLSTM,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)


def loss_fn(loss):
    loss_list = {
        "l1": nn.L1Loss(),
        "l2": nn.MSELoss(reduction="mean")
    }
    return loss_list[loss]


def inverse_y_values(predictions, values, df_test, scalar):
    values = np.concatenate(values, axis=0).ravel()
    predictions = np.concatenate(predictions, axis=0).ravel()
    df = pd.DataFrame(data={"y_test": values, "y_hat": predictions}, index=df_test.head(len(values)).index)
    # sort the data by year-date-time
    df = df.sort_index()
    # inverse the value by the scalar that we used to normalize the data
    for col in [["y_test", "y_hat"]]:
        df[col] = scalar.inverse_transform(df[col])
    return df


def calculate_metrics(df):
    return {'mae': mean_absolute_error(df.y_test, df.y_hat),
            'rmse': mean_squared_error(df.y_test, df.y_hat) ** 0.5,
            'r2': r2_score(df.y_test, df.y_hat)}


def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}': lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period),
        f'cos_{col_name}': lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    }
    return df.assign(**kwargs).drop(columns=[col_name])


def remove_outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range
