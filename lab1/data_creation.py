import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Создание дневной температуры со случайными аномалиями
def generate_temperature_data(start_date, end_date, anomalies_prob=0.1):
    dates = pd.date_range(start=start_date, end=end_date)
    temperature = np.random.normal(25, 5, len(dates))  # Среднее 25°C, стандартное отклонение 5°C

    # Добавление аномалий
    anomalies = np.random.choice([True, False], size=len(dates), p=[anomalies_prob, 1 - anomalies_prob])
    temperature[anomalies] += np.random.normal(0, 15, np.sum(anomalies))  # Случайное изменение температуры

    return dates, temperature


# График температуры
def plot_temperature_data(dates, temperature, title):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, temperature, color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Создание и сохранение наборов данных
def create_and_save_datasets(num_datasets, data_dir, start_date, end_date, anomalies_prob=0.1):
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_datasets):
        dates, temperature = generate_temperature_data(start_date, end_date, anomalies_prob)
        title = f'Dataset {i+1}'
        plot_temperature_data(dates, temperature, title)
        np.savetxt(os.path.join(data_dir, f'dataset_{i+1}.csv'), np.vstack((dates.astype(str), temperature)).T, delimiter=',', fmt='%s', header='Date,Temperature', comments='')


# Параметры
start_date = '2023-01-01'
end_date = '2023-12-31'
num_train_datasets = 5
num_test_datasets = 2
train_data_dir = 'train'
test_data_dir = 'test'


# Создание и сохранение наборов данных для тренировки и тестирования
create_and_save_datasets(num_train_datasets, train_data_dir, start_date, end_date)
create_and_save_datasets(num_test_datasets, test_data_dir, start_date, end_date)