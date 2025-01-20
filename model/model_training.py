import numpy as np
import matplotlib.pyplot as plt
from data.data_preprocessing import add_single_colomn
import time  # Для замера времени

# Инициализация весов (theta)
def initialize_theta(n):
    return np.zeros(n)  # Создаём массив из нулей размером n (количество признаков)

# Функция для предсказания
def predict(X, theta):
    return np.dot(X, theta)  # Вычисляем предсказания: X * theta

# Функция стоимости (Среднеквадратичная ошибка, MSE)
def compute_cost(X, y, theta):
    m = len(y)  # Количество объектов
    predictions = predict(X, theta)  # Вычисляем предсказания
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # Формула MSE
    return cost

# Градиентный спуск
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)  # Количество объектов
    cost_history = []  # Список для хранения значений функции стоимости на каждой итерации

    for i in range(iterations):
        predictions = predict(X, theta)  # Вычисляем предсказания
        gradients = (1 / m) * np.dot(X.T, (predictions - y))  # Вычисляем градиенты

        # Проверка на слишком большие градиенты
        if np.any(np.abs(gradients) > 1e10):
            print(f"Градиенты слишком большие на итерации {i}, остановка обучения.")
            break

        theta -= alpha * gradients  # Обновляем веса
        cost = compute_cost(X, y, theta)  # Вычисляем текущую стоимость
        cost_history.append(cost)  # Сохраняем стоимость

        # Каждую сотую итерацию выводим информацию
        if i % 100 == 0 or i == iterations - 1:
            print(f"Итерация {i}: Стоимость = {cost:.6f}")

    return theta, cost_history

# Обучение модели
def train_model(X_train, y_train, alphas, iterations):
    costs = {}  # Словарь для хранения истории функции стоимости для каждого значения alpha
    best_alpha = None  # Лучшее значение alpha
    best_cost = float('inf')  # Лучшее значение функции стоимости (инициализируем как бесконечность)
    best_theta = None  # Лучшие параметры (веса)

    for alpha in alphas:
        print(f"\nОбучение с alpha={alpha}")
        theta = initialize_theta(X_train.shape[1])  # Инициализация весов

        # Замер времени начала обучения
        start_time = time.time()

        theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, iterations)  # Градиентный спуск

        # Замер времени окончания обучения
        end_time = time.time()

        training_time = end_time - start_time  # Расчет времени обучения
        print(f"Время обучения с alpha={alpha}: {training_time:.2f} секунд")

        costs[alpha] = cost_history  # Сохраняем историю функции стоимости

        # Проверяем, является ли текущее значение функции стоимости наименьшим
        if cost_history and cost_history[-1] < best_cost:
            best_cost = cost_history[-1]  # Обновляем лучшее значение функции стоимости
            best_alpha = alpha  # Обновляем лучшее значение alpha
            best_theta = theta  # Обновляем лучшие параметры

    return best_theta, best_alpha, best_cost, costs

if __name__ == "__main__":
    # Загрузка обучающих данных
    X_train = np.loadtxt('../data/X_train.txt', delimiter=',')  # Загружаем обучающие признаки из файла
    y_train = np.loadtxt('../data/y_train.txt', delimiter=',')  # Загружаем обучающую целевую переменную из файла

    # Добавляем столбец единиц для bias term
    X_train = add_single_colomn(X_train)

    alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0, 1.1, 1.2]  # Список значений alpha (скорость обучения)
    iterations = 1000  # Количество итераций

    # Замер времени начала общего процесса обучения
    overall_start_time = time.time()

    # Обучение модели
    best_theta, best_alpha, best_cost, costs = train_model(X_train, y_train, alphas, iterations)

    # Замер времени окончания общего процесса обучения
    overall_end_time = time.time()

    # Вычисление общего времени обучения
    overall_training_time = overall_end_time - overall_start_time
    print(f"\nОбщее время обучения: {overall_training_time:.2f} секунд")

    # Сохраняем лучшие параметры в файл
    np.savetxt('optimized_theta.csv', best_theta, delimiter=',', header='theta', comments='')

    # Построение графика изменения функции стоимости для каждого значения alpha
    plt.figure(figsize=(10, 6))  # Размер графика
    for alpha, cost_history in costs.items():
        plt.plot(range(1, len(cost_history) + 1), cost_history, label=f'alpha={alpha}')

    plt.xlabel('Количество итераций')  # Подпись оси X
    plt.ylabel('Функция стоимости (J)')  # Подпись оси Y
    plt.title('Сходимость градиентного спуска при разных значениях alpha')  # Заголовок графика
    plt.legend()  # Отображение легенды
    plt.ylim(0, 1000)  # Ограничение для оси Y
    plt.show()  # Отображение графика

    # Вывод лучшего значения alpha и соответствующей функции стоимости
    print(f"\nЛучшее значение alpha: {best_alpha} с финальной функцией стоимости: {best_cost:.6f}")
