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
    # Загрузка тестовых данных
    X_test = np.loadtxt('../data/X_test.txt', delimiter=',')  # Тестовые признаки
    y_test = np.loadtxt('../data/y_test.txt', delimiter=',')  # Тестовая целевая переменная
    theta = np.loadtxt('optimized_theta.csv', delimiter=',', skiprows=1)  # Параметры обученной модели

    # Предсказания для тестовой выборки
    predictions = predict(X_test, theta)

    # Оценка метрик на тестовой выборке
    mae, mse, mape, r_squared = evaluate_model(predictions, y_test)

    # Вывод результатов оценки
    print("Результаты оценки модели на тестовых данных:")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Коэффициент детерминации (R-squared): {r_squared:.4f}")
    print(f"Средний абсолютный процент ошибки (MAPE): {mape:.2f}%")
