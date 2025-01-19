import numpy as np
from data.data_preprocessing import normalize_features, add_zeros_colomn
from model_training import predict

def load_min_max():
    feature_min = np.loadtxt('../data/feature_min.txt', delimiter=',')
    feature_max = np.loadtxt('../data/feature_max.txt', delimiter=',')
    return feature_min, feature_max

if __name__ == "__main__":
    # Загрузка минимальных и максимальных значений
    feature_min, feature_max = load_min_max()
    theta = np.loadtxt('optimized_theta.csv', delimiter=',', skiprows=1)

    # Пример пользовательского ввода
    user_input = [0,2.4,4,25,4,11.2,7.7,9.6,29]
    normalized_input = normalize_features(np.array(user_input).reshape(1, -1), feature_min, feature_max)
    normalized_input = add_zeros_colomn(normalized_input)

    # Проверка на размерность и добавление веса для bias, если необходимо
    if theta.shape[0] + 1 == normalized_input.shape[1]:
        theta = np.insert(theta, 0, 0)  # Добавляем нулевой вес для bias

    # Предсказание
    user_prediction = predict(normalized_input, theta)
    print(f"Прогнозируемый выброс CO2 для пользовательского ввода: {user_prediction[0]}")
