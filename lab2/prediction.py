import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Загрузка модели из файла
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Загрузка тестовых данных из CSV файла
df_test = pd.read_csv('iris_test_preprocessed.csv')

# Разделение на признаки и метки классов
X_test = df_test.drop('target', axis=1)
y_test = df_test['target']

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Оценка качества модели на тестовых данных
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели на тестовых данных: {accuracy:.2f}")