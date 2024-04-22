import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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


def train_model(train_data):
    X_train = np.array([])
    y_train = np.array([])
    for df in train_data:
        X_train = np.append(X_train, df['Date'].values)  # Используем дату в качестве признака
        y_train = np.append(y_train, df['Temperature'].values)

    X_train = X_train.reshape(-1, 1)  # Преобразуем в двумерный массив
    y_train = y_train.reshape(-1, 1)

    # Создаем и обучаем модель случайного леса
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model


def save_model(model, model_path):
    joblib.dump(model, model_path)


def main(train_data_dir, model_output_path):
    # Загрузка данных
    train_data = load_data(train_data_dir)

    # Предобработка данных
    train_data = preprocess_data(train_data)

    # Обучение модели
    model = train_model(train_data)

    # Сохранение модели
    save_model(model, model_output_path)
    print("Модель успешно обучена и сохранена.")


# Обучаем данные
if __name__ == "__main__":
    train_data_dir = "preprocessed_train"
    model_output_path = "trained_model.pkl"
    main(train_data_dir, model_output_path)