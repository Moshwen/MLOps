import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Загрузка данных из CSV файла
df_train = pd.read_csv('iris_train_preprocessed.csv')

# Разделение на признаки и метки классов
X_train = df_train.drop('target', axis=1)
y_train = df_train['target']

# Создание и обучение модели машинного обучения (логистическая регрессия)
model = LogisticRegression()
model.fit(X_train, y_train)

# Сохранение модели в файл с помощью pickle
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)