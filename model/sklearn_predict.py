import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Загрузка данных
data = pd.read_excel('../data/co2.xlsx')  # Исходные данные

# Приведение всех категориальных столбцов к строковому типу
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype(str)

# Разделение на признаки (X) и целевую переменную (y)
X = data.drop('CO2 Emissions(g/km)', axis=1)
y = data['CO2 Emissions(g/km)']

# Определение числовых и категориальных столбцов
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Применяем обработку пропусков
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Или MinMaxScaler для нормализации
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncoder для категорий
])

# Объединение всех преобразований в один ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Создание модели с обработкой данных
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование
y_pred = model.predict(X_test)

# Оценка модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R^2: {r2}")
print(f"Средний абсолютный процент ошибки (MAPE): {mape:.2f}%")

# Пример новых данных для предсказания
new_data = pd.DataFrame({
    'Make': ['ACURA'],
    'Model': ['ILX HYBRID'],
    'Vehicle Class': ['COMPACT'],
    'Engine Size(L)': [1.5],
    'Cylinders': [4],
    'Transmission': ['AV7'],
    'Fuel Type': ['Z'],
    'Fuel Consumption City (L/100 km)': [6],
    'Fuel Consumption Hwy (L/100 km)': [5.8],
    'Fuel Consumption Comb (L/100 km)': [5.9],
    'Fuel Consumption Comb (mpg)': [48]
})

# Приведение новых данных к строковому типу (если есть категориальные столбцы)
for col in new_data.select_dtypes(include=['object']).columns:
    new_data[col] = new_data[col].astype(str)

# Прогнозирование с использованием ранее обученной модели
y_pred_new = model.predict(new_data)

# Выводим результат
print(f"Прогнозируемое значение выбросов CO2: {y_pred_new[0]}")
