## Практическая работа 1
Набор данных представляет собой средняя температура воздуха для каждого дня 2023 года. Были созданы 5 наборов данных для обучения и 2 набора для тестирования. В каждом наборе были добавлены шумы и аномалии, которые были убраны в этапе предварительной обработки. 
<p>В ходе предварительной обработки все даты были преобразованы с помощью библиотеки pandas на чило дней от начала года. Для обучения модели был использован модель случайного леса из библиотеки <i>scikit-learn</i>.</p>

Для коректной работы программы необходимо загрузить следующие библиотеки :
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error   
```
