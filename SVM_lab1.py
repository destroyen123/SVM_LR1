# Лабораторная работа № 1. Вариант 3. Куликов Е.В. M30-237БВ-25
import numpy as np

np.set_printoptions(precision=8, suppress=True)

# Исходная матрица коэффициентов.
A = np.array([
    [11.72, -2.15, 1.26, 0.91],
    [1.33, 12.84, -1.48, 2.17],
    [-1.74, 1.21, 10.96, 1.58],
    [2.03, -1.16, 0.87, 13.41]
])

b = np.array([12.84, 19.63, 13.47, 21.35])

print("Матрица коэффициентов A:")
print(A)

print("\nВектор правой части b:")
print(b)

# Контрольное решение средствами NumPy.
x_lib = np.linalg.solve(A, b)

det = np.linalg.det(A)
cond = np.linalg.cond(A)

print("\nРешение системы средствами NumPy:")
print(x_lib)

print("\nОпределитель матрицы:")
print(det)

print("\nЧисло обусловленности:")
print(cond)

# Метод Гаусса без выбора ведущего элемента.
def gauss(A, b):
    A = A.copy()
    b = b.copy()
    n = len(b)

    # Прямой ход: исключение неизвестных.
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    print("\nМатрица после прямого хода:")
    print(A)

    # Обратный ход.
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x


x_gauss = gauss(A, b)

print("\nРешение методом Гаусса:")
print(x_gauss)


# Метод Гаусса с выбором ведущего элемента.
def gauss_pivot(A, b):
    A = A.copy()
    b = b.copy()
    n = len(b)

    for k in range(n - 1):
        # Поиск максимального по модулю элемента.
        p = k + np.argmax(np.abs(A[k:, k]))

        # Перестановка строк матрицы и правой части.
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    # Обратный ход.
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x


x_pivot = gauss_pivot(A, b)

print("\nРешение методом Гаусса с выбором ведущего элемента:")
print(x_pivot)

# Построение итерационной матрицы.
D = np.diag(A)
B = -(A - np.diag(D)) / D[:, None]

# Проверка условий сходимости.
dominance = np.abs(D) > (
    np.sum(np.abs(A), axis=1) - np.abs(D)
)

rho = np.max(np.abs(np.linalg.eigvals(B)))

print("\nПроверка диагонального преобладания:")
print(dominance)

print("\nСпектральный радиус:")
print(rho)

# Метод простой итерации (Якоби).
def jacobi(A, b, x0, eps=1e-6, max_iter=1000):
    x = x0.copy()
    n = len(b)

    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)

        # Используем значения предыдущего приближения.
        for i in range(n):
            s = sum(A[i, j] * x[j]
                    for j in range(n) if j != i)

            x_new[i] = (b[i] - s) / A[i, i]

        # Оценка изменения решения.
        error = np.linalg.norm(x_new - x)

        print(f"Итерация {k}: {x_new}, ошибка = {error:.8e}")

        if error < eps:
            return x_new, k

        x = x_new

    print("Требуемая точность не достигнута")
    return x, max_iter


print("\nМетод простой итерации:")

x_jacobi, k_jacobi = jacobi(A, b, np.zeros(len(b)))

print("\nРешение методом простой итерации:")
print(x_jacobi)

print("Количество итераций:", k_jacobi)

# Метод Зейделя.
def seidel(A, b, x0, eps=1e-6, max_iter=1000):
    x = x0.copy()
    n = len(b)

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            # Используем обновлённые и предыдущие компоненты.
            s1 = A[i, :i] @ x[:i]
            s2 = A[i, i+1:] @ x_old[i+1:]

            x[i] = (b[i] - s1 - s2) / A[i, i]

        # Проверка точности приближения.
        error = np.linalg.norm(x - x_old)

        print(f"Итерация {k}: {x}, ошибка = {error:.8e}")

        if error < eps:
            return x, k

    print("Требуемая точность не достигнута")
    return x, max_iter


print("\nМетод Зейделя:")

x_seidel, k_seidel = seidel(A, b, np.zeros(len(b)))

print("\nРешение методом Зейделя:")
print(x_seidel)

print("Количество итераций:", k_seidel)

# Вычисление вектора невязки.
def residual(A, b, x):
    return A @ x - b


# Абсолютное и относительное отклонения от NumPy.
def compare_solution(x, x_lib):
    absolute = np.linalg.norm(x - x_lib)
    relative = absolute / np.linalg.norm(x_lib)

    return absolute, relative


# Сравнение полученных решений.
methods = {
    "NumPy": x_lib,
    "Метод Гаусса": x_gauss,
    "Метод Гаусса с выбором элемента": x_pivot,
    "Метод простой итерации": x_jacobi,
    "Метод Зейделя": x_seidel
}

print("\nСравнение результатов методов:")

for name, x in methods.items():
    r = residual(A, b, x)
    absolute, relative = compare_solution(x, x_lib)

    print(f"\n{name}")
    print("Решение:", x)
    print("Вектор невязки:", r)
    print(f"Норма невязки: {np.linalg.norm(r):.8e}")
    print(f"Абсолютная ошибка: {absolute:.8e}")
    print(f"Относительная ошибка: {relative:.8e}")

print("\n15. Вычислительный эксперимент с начальными приближениями")

# Различные начальные приближения.
initial_vectors = [
    np.zeros(4),
    np.ones(4),
    np.array([5.0, -3.0, 2.0, 4.0])
]

results_initial = []

for x0 in initial_vectors:
    for name, method in [
        ("Метод простой итерации", jacobi),
        ("Метод Зейделя", seidel)
    ]:
        print(f"\n{name}")
        print("Начальное приближение:", x0)

        x, k = method(A, b, x0, eps=1e-6)
        r_norm = np.linalg.norm(residual(A, b, x))

        results_initial.append((name, x0, x, k, r_norm))

print("\nИтоги эксперимента:")

for name, x0, x, k, r_norm in results_initial:
    print(f"\n{name}")
    print("Начальное приближение:", x0)
    print("Решение:", x)
    print("Количество итераций:", k)
    print(f"Норма невязки: {r_norm:.8e}")

print("\n16. Вычислительный эксперимент с различной точностью")

# Исследование влияния заданной точности.
eps_values = [1e-2, 1e-4, 1e-6]
results_eps = []

for eps in eps_values:
    for name, method in [
        ("Метод простой итерации", jacobi),
        ("Метод Зейделя", seidel)
    ]:
        print(f"\n{name}, точность = {eps}")

        x, k = method(A, b, np.zeros(4), eps=eps)
        r_norm = np.linalg.norm(residual(A, b, x))

        results_eps.append((name, eps, x, k, r_norm))

print("\nИтоги эксперимента с различной точностью:")

for name, eps, x, k, r_norm in results_eps:
    print(f"\n{name}")
    print(f"Точность: {eps:.0e}")
    print("Решение:", x)
    print("Количество итераций:", k)
    print(f"Норма невязки: {r_norm:.8e}")

print("\n17. Исследование влияния округления коэффициентов")

# Округляем матрицу, не изменяя правую часть.
for decimals in [0, 1, 2, 4]:
    A_round = np.round(A, decimals)

    x_round = gauss_pivot(A_round, b)

    delta = np.linalg.norm(x_lib - x_round)
    cond_round = np.linalg.cond(A_round)

    print(f"\nОкругление до {decimals} знаков")
    print("Округленная матрица:")
    print(A_round)
    print("Решение:", x_round)
    print(f"Изменение решения: {delta:.8e}")
    print(f"Число обусловленности: {cond_round:.8f}")

print("\nИсходное решение:", x_lib)
print("Исходное число обусловленности:", cond)

print("\n18. Сравнение прямого и итерационных методов")

# Итоговое сравнение методов.
iterations = {
    "NumPy": "-",
    "Метод Гаусса": "-",
    "Метод Гаусса с выбором элемента": "-",
    "Метод простой итерации": k_jacobi,
    "Метод Зейделя": k_seidel
}

print(f"{'Метод':40} {'Итерации':>10} {'Невязка':>15} {'Ошибка':>15}")

for name, x in methods.items():
    r_norm = np.linalg.norm(residual(A, b, x))
    absolute, relative = compare_solution(x, x_lib)

    print(f"{name:40} {iterations[name]:>10} "
          f"{r_norm:15.8e} {absolute:15.8e}")

print("\nВычислительная сложность:")
print("Метод Гаусса: O(n^3)")
print("Обратный ход Гаусса: O(n^2)")
print("Метод простой итерации: O(k*n^2)")
print("Метод Зейделя: O(k*n^2)")

print("\n19. Проверка собственной реализации")

# Проверка совпадения с контрольным решением.
for name, x in methods.items():
    if name == "NumPy":
        continue

    r_norm = np.linalg.norm(residual(A, b, x))
    absolute, relative = compare_solution(x, x_lib)

    print(f"\n{name}")
    print(f"Норма невязки: {r_norm:.8e}")
    print(f"Отклонение от NumPy: {absolute:.8e}")

    if np.allclose(x, x_lib, atol=1e-6, rtol=0):
        print("Проверка пройдена")
    else:
        print("Проверка не пройдена")
