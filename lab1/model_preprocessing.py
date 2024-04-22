import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Функция для загрузки данных из CSV-файла
def load_data(data_dir):
    data = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            data.append(df)
    return data


data = load_data("train")


# Функция для предобработки данных с помощью StandardScaler
def preprocess_data(data):
    scaler = StandardScaler()
    preprocessed_data = []
    for df in data:
        # Преобразование признаков (температуры)
        temperature = df['Temperature'].values.reshape(-1, 1)
        temperature_scaled = scaler.fit_transform(temperature)

        # Обновление DataFrame с масштабированными значениями температуры
        df['Temperature'] = temperature_scaled.flatten()
        preprocessed_data.append(df)
    return preprocessed_data


# Функция для сохранения предобработанных данных в CSV-файлы
def save_preprocessed_data(preprocessed_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, df in enumerate(preprocessed_data):
        file_path = os.path.join(output_dir, f'preprocessed_dataset_{i+1}.csv')
        df.to_csv(file_path, index=False)


# Пути к данным
train_data_dir = 'train'
test_data_dir = 'test'
output_train_data_dir = 'preprocessed_train'
output_test_data_dir = 'preprocessed_test'


# Загрузка данных
train_data = load_data(train_data_dir)
test_data = load_data(test_data_dir)


# Предобработка данных
preprocessed_train_data = preprocess_data(train_data)
preprocessed_test_data = preprocess_data(test_data)


# Сохранение предобработанных данных
save_preprocessed_data(preprocessed_train_data, output_train_data_dir)
save_preprocessed_data(preprocessed_test_data, output_test_data_dir)