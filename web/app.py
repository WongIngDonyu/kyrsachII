from flask import Flask, request, render_template
import numpy as np
import json
from data.data_preprocessing import normalize_features, add_zeros_colomn
from model.model_training import predict


# Функция для загрузки минимальных и максимальных значений
def load_min_max():
    feature_min = np.loadtxt('../data/feature_min.txt', delimiter=',')
    feature_max = np.loadtxt('../data/feature_max.txt', delimiter=',')
    return feature_min, feature_max


# Функция для загрузки словарей кодирования категориальных данных
def load_encoders():
    with open('../data/encoders.json', 'r') as file:
        return json.load(file)


# Создание веб-приложения
app = Flask(__name__)

# Загружаем параметры модели, минимумы/максимумы и словари кодирования
feature_min, feature_max = load_min_max()
encoders = load_encoders()
theta = np.loadtxt('../model/optimized_theta.csv', delimiter=',', skiprows=1)

# Если необходимо, добавляем нулевой вес для bias
if theta.shape[0] + 1 == len(feature_min) + 1:
    theta = np.insert(theta, 0, 0)  # Добавляем нулевой вес для bias


# Функция для кодирования категориальных данных
def encode_categorical(user_input, encoders):
    encoded_input = []
    for key, value in user_input.items():
        if key in encoders:  # Если поле является категориальным
            if value in encoders[key]:
                encoded_input.append(encoders[key][value])  # Кодируем значение
            else:
                raise ValueError(f"Invalid value '{value}' for field '{key}'.")
        else:
            encoded_input.append(float(value))  # Для числовых данных
    return encoded_input


# Главная страница
@app.route('/')
def home():
    return render_template('index.html')


# Обработка ввода пользователя
@app.route('/predict', methods=['POST'])
def predict_value():
    try:
        # Получение данных из формы
        user_input = {
            'Vehicle Class': request.form['vehicle_class'],
            'Engine Size(L)': request.form['engine_size'],
            'Cylinders': request.form['cylinders'],
            'Transmission': request.form['transmission'],
            'Fuel Type': request.form['fuel_type'],
            'Fuel Consumption City (L/100 km)': request.form['fuel_consumption_city'],
            'Fuel Consumption Hwy (L/100 km)': request.form['fuel_consumption_hwy'],
            'Fuel Consumption Comb (L/100 km)': request.form['fuel_consumption_comb'],
            'Fuel Consumption Comb (mpg)': request.form['fuel_consumption_comb_mpg']
        }

        # Кодирование категориальных данных и подготовка данных
        encoded_input = encode_categorical(user_input, encoders)

        # Нормализация данных пользователя
        normalized_input = normalize_features(np.array(encoded_input).reshape(1, -1), feature_min, feature_max)
        normalized_input = add_zeros_colomn(normalized_input)

        # Предсказание
        user_prediction = predict(normalized_input, theta)

        # Возврат предсказания и введённых данных на страницу
        return render_template(
            'index.html',
            prediction=f"Прогнозируемый выброс CO2: {user_prediction[0]:.2f}",
            user_input=user_input
        )

    except Exception as e:
        # Возврат ошибки и введённых данных на страницу
        return render_template(
            'index.html',
            prediction=f"Error: {str(e)}",
            user_input=request.form
        )

if __name__ == '__main__':
    app.run(debug=True)
