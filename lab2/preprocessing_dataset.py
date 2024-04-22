import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Выбор важных признаков с помощью метода анализа дисперсии (ANOVA)
k_best_features = 2
selector = SelectKBest(score_func=f_classif, k=k_best_features)
X_new = selector.fit_transform(X_scaled, y)

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Сохранение данных в DataFrame и CSV
df_train = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1,1))), columns=[f"feature_{i}" for i in range(k_best_features)] + ['target'])
df_test = pd.DataFrame(np.hstack((X_test, y_test.reshape(-1,1))), columns=[f"feature_{i}" for i in range(k_best_features)] + ['target'])

# Сохранение в CSV
df_train.to_csv('iris_train_preprocessed.csv', index=False)
df_test.to_csv('iris_test_preprocessed.csv', index=False)
