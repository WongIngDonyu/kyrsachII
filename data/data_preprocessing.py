import numpy as np
import pandas as pd
import json

# Загрузка датасета
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Кодирование категориальных переменных
def encode_categorical(data, columns):
    encoders = {}
    for col in columns:
        unique_values = sorted(data[col].unique())  # Сортируем значения для одинакового порядка
        encoders[col] = {val: idx for idx, val in enumerate(unique_values)}  # Создаём словарь кодирования
        data[col] = data[col].map(encoders[col])  # Применяем кодирование
    return data, encoders

# Сохранение словарей кодирования в JSON файл
def save_encoders(encoders, output_file):
    with open(output_file, 'w') as file:
        json.dump(encoders, file, indent=4)

# Добавление столбца нулей к данным
def add_zeros_colomn(X):
    return np.c_[np.ones(X.shape[0]), X]

# Вычисление минимальных и максимальных значений признаков
def calculate_min_max(X):
    feature_min = np.min(X, axis=0)  # Минимальные значения по каждому признаку
    feature_max = np.max(X, axis=0)  # Максимальные значения по каждому признаку
    return feature_min, feature_max

# Нормализация признаков
def normalize_features(X, feature_min, feature_max):
    return (X - feature_min) / (feature_max - feature_min)  # Нормализация по формуле Min-Max

# Сохранение нормализованных данных
def save_normalized_data(X, y, output_file):
    # Создаем DataFrame для нормализованных данных
    data_normalized = pd.DataFrame(X, columns=data_cleaned.drop(columns=['CO2 Emissions(g/km)']).columns)
    # Добавляем нормализованную целевую переменную
    data_normalized['CO2 Emissions(g/km)'] = normalize_features(y.reshape(-1, 1), np.min(y), np.max(y)).flatten()
    # Сохраняем в Excel файл
    data_normalized.to_excel(output_file, index=False)

if __name__ == "__main__":
    # Загружаем исходные данные
    data = load_data('co2.xlsx')
    # Удаляем ненужные столбцы
    data_cleaned = data.drop(columns=['Make', 'Model'])

    # Сохраняем очищенные данные
    data_cleaned.to_excel('cleaned_data.xlsx', index=False)

    # Обрабатываем категориальные переменные
    categorical_columns = ['Vehicle Class', 'Transmission', 'Fuel Type']
    data_cleaned, label_encoders = encode_categorical(data_cleaned, categorical_columns)

    # Сохраняем данные с закодированными категориальными переменными
    data_cleaned.to_excel('encoded_data.xlsx', index=False)

    # Сохраняем словари кодирования в JSON файл
    save_encoders(label_encoders, 'encoders.json')

    # Разделяем данные на признаки (X) и целевую переменную (y)
    X = data_cleaned.drop(columns=['CO2 Emissions(g/km)']).values
    y = data_cleaned['CO2 Emissions(g/km)'].values

    # Вычисляем минимальные и максимальные значения признаков
    feature_min, feature_max = calculate_min_max(X)
    # Нормализуем данные
    X = normalize_features(X, feature_min, feature_max)

    # Сохраняем нормализованные данные
    save_normalized_data(X, y, 'normalized_data.xlsx')

    # Сохраняем минимальные и максимальные значения признаков
    np.savetxt('feature_min.txt', feature_min, delimiter=',')
    np.savetxt('feature_max.txt', feature_max, delimiter=',')

    # Сохраняем признаки (X) и целевую переменную (y) в текстовые файлы
    np.savetxt('X.txt', X, delimiter=',')
    np.savetxt('y.txt', y, delimiter=',')
