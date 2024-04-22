from sklearn.datasets import load_iris

# Загрузка игрового датасета Iris
iris = load_iris()

# Вывод информации о датасете
print("Информация о датасете Iris:")
print("Названия признаков:")
print(iris.feature_names)
print("Названия классов:")
print(iris.target_names)
print("Размерность данных:")
print(iris.data.shape)
print("Первые 5 строк данных:")
print(iris.data[:5])
print("Первые 5 меток классов:")
print(iris.target[:5])