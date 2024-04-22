import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


def load_data(data_dir):
    data = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            data.append(df)
    return data


def preprocess_data(data):
    for df in data:
        # Преобразуем дату в числовой формат (например, количество дней с начала года)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = (df['Date'] - df['Date'].min()).dt.days
    return data


def load_model(model_path):
    return joblib.load(model_path)


def test_model(test_data, model):
    predictions = []
    actual_values = []
    for df in test_data:
        X_test = df['Date'].values.reshape(-1, 1)  # Используем дату в качестве признака
        y_test = df['Temperature'].values

        # Прогнозирование температуры
        y_pred = model.predict(X_test)

        # Сохранение предсказанных и фактических значений
        predictions.append(y_pred)
        actual_values.append(y_test)

    return predictions, actual_values


def evaluate(predictions, actual_values):
    mse_scores = []
    for pred, actual in zip(predictions, actual_values):
        mse = mean_squared_error(actual, pred)
        mse_scores.append(mse)

    return mse_scores


def main(test_data_dir, model_path):
    # Загрузка данных
    test_data = load_data(test_data_dir)

    # Предобработка данных
    test_data = preprocess_data(test_data)

    # Загрузка модели
    model = load_model(model_path)

    # Тестирование модели
    predictions, actual_values = test_model(test_data, model)

    # Оценка результатов
    mse_scores = evaluate(predictions, actual_values)
    for i, mse in enumerate(mse_scores):
        print(f"Dataset {i+1} MSE: {mse}")


if __name__ == "__main__":
    test_data_dir = "preprocessed_test"
    model_path = "trained_model.pkl"
    main(test_data_dir, model_path)