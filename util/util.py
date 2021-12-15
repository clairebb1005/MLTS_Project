from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.LSTM import LSTMModel
from models.GRU import GRUModel
from models.BiLSTM import BiLSTM
import numpy as np
import pandas as pd


def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y


def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model(model, model_params):
    models = {
        "lstm": LSTMModel,
        "bilstm": BiLSTM,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)


def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


def calculate_metrics(df):
    return {'mae': mean_absolute_error(df.value, df.prediction),
            'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2': r2_score(df.value, df.prediction)}
