import numpy as np
from model_training import predict

# Оценка качества модели
def evaluate_model(predictions, y):
    # Средняя абсолютная ошибка (MAE)
    mae = np.mean(np.abs(predictions - y))
    # Среднеквадратичная ошибка (MSE)
    mse = np.mean((predictions - y) ** 2)
    # Средний абсолютный процент ошибки (MAPE)
    mape = np.mean(np.abs((y - predictions) / y)) * 100
    # Общая сумма квадратов
    ss_total = np.sum((y - np.mean(y)) ** 2)
    # Остаточная сумма квадратов
    ss_residual = np.sum((y - predictions) ** 2)
    # Коэффициент детерминации (R-squared)
    r_squared = 1 - (ss_residual / ss_total)
    return mae, mse, mape, r_squared

if __name__ == "__main__":
    # Загрузка данных
    X = np.loadtxt('../data/X.txt', delimiter=',')  # Признаки
    y = np.loadtxt('../data/y.txt', delimiter=',')  # Целевая переменная
    theta = np.loadtxt('optimized_theta.csv', delimiter=',', skiprows=1)  # Параметры модели

    # Предсказания
    predictions = predict(X, theta)

    # Оценка метрик
    mae, mse, mape, r_squared = evaluate_model(predictions, y)

    # Вывод результатов
    print(f"Средняя абсолютная ошибка (MAE): {mae}")
    print(f"Среднеквадратичная ошибка (MSE): {mse}")
    print(f"Коэффициент детерминации (R-squared): {r_squared}")
    print(f"Средний абсолютный процент ошибки (MAPE): {mape:.2f}%")
