import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import scipy.linalg
from scipy.linalg import lu
from scipy.linalg import solve_triangular


# Zadanie A - Tworzenie układu równań
def create_equation_system(N, a1, a2, a3):
    # Tworzenie macierzy A
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = a1
        if i == 0:
            A[i, i + 1] = a2
            A[i, i + 2] = a3
        elif i == 1:
            A[i, i + 1] = a2
            A[i, i + 2] = a3
            A[i, i - 1] = a2
        elif i == N - 2:
            A[i, i - 1] = a2
            A[i, i - 2] = a3
            A[i, i + 1] = a2
        elif i == N - 1:
            A[i, i - 1] = a2
            A[i, i - 2] = a3
        else:
            A[i, i - 1] = a2
            A[i, i - 2] = a3
            A[i, i + 1] = a2
            A[i, i + 2] = a3

    # Tworzenie wektora b

    b = np.array([np.sin(n * 4) for n in range(1, N + 1)])

    return A, b


# Zadanie B - Implementacja metody Jacobiego
def jacobi_iteration(A, b, max_iterations=1000, tol=1e-9):
    N = len(A)

    # Inicjalizacja wektora x wartościami początkowymi
    x = np.ones(N)

    # Inicjalizacja macierzy M i wektora bm
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    M = -np.linalg.inv(D) @ (L + U)
    bm = np.linalg.inv(D) @ b

    # Inicjalizacja listy norm residuum
    residuals = []

    # Iteracyjne rozwiązywanie równania macierzowego metodą Jacobiego
    for iterations in range(max_iterations):
        x_next = M @ x + bm
        err_norm = np.linalg.norm(A @ x_next - b)
        residuals.append(err_norm)  # Dodanie normy residuum do listy
        if err_norm < tol:
            break
        x = x_next
    return x, residuals


# Zadanie B - Implementacja metody Gaussa-Seidla
def gauss_seidel_iteration(A, b, max_iterations=1000, tol=1e-9):
    N = len(A)

    # Inicjalizacja macierzy D, L, U, M i wektora bm
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    M = -(np.linalg.inv(D + L) @ U)
    bm = np.linalg.inv(D + L) @ b

    # Inicjalizacja wektora x wartościami początkowymi
    x = np.ones(N)

    # Inicjalizacja listy norm residuum
    residuals = []

    # Iteracyjne rozwiązywanie równania macierzowego metodą Gaussa-Seidela
    for iterations in range(max_iterations):
        x_next = M @ x + bm
        err_norm = np.linalg.norm(A @ x_next - b)
        residuals.append(err_norm)  # Dodanie normy residuum do listy
        if err_norm < tol:
            break
        x = x_next
    return x, residuals


# Zadanie D - Implementacja metody faktoryzacji LU
def lu_factorization(A, b):
    N = len(A)  # Wymiary macierzy kwadratowej

    # Inicjalizacja macierzy L i U
    L = [[0.0] * N for _ in range(N)]
    U = [[0.0] * N for _ in range(N)]
    x = [0.0] * N

    # Faktoryzacja LU
    for i in range(N):
        # Rozwiązanie dla L
        for j in range(i + 1):
            s1 = sum(U[k][i] * L[j][k] for k in range(j))
            L[i][j] = A[i][j] - s1
        # Rozwiązanie dla U
        for j in range(i, N):
            s2 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = (A[i][j] - s2) / L[i][i]

    # Rozwiązanie Ly = b
    y = [0.0] * N
    for i in range(N):
        s3 = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s3) / L[i][i]

    # Rozwiązanie Ux = y
    for i in range(N - 1, -1, -1):
        s4 = sum(U[i][j] * x[j] for j in range(i + 1, N))
        x[i] = (y[i] - s4) / U[i][i]

    # Obliczenie residuum
    residual = sum((sum(A[i][j] * x[j] for j in range(N)) - b[i]) ** 2 for i in range(N)) ** 0.5

    return x, residual


# Parametry
N = 955
a1 = 12
a2 = a3 = -1

# Zadanie A - Tworzenie układu równań dla podanych parametrów
A, b = create_equation_system(N, a1, a2, a3)

print("Macierz A:")
print(A)

# Wyświetlenie wektora b
print("\nWektor b:")
print(b)

# Zadanie B - Rozwiązanie układu równań dla zadania A metodami iteracyjnymi
start_time = time.time()
x_jacobi, residuals_jacobi = jacobi_iteration(A, b)
end_time = time.time()
execution_jacobi_time = end_time - start_time
print("Czas wykonania Jacobi:", execution_jacobi_time, "sekund")

start_time = time.time()
x_gauss_seidel, residuals_gauss_seidel = gauss_seidel_iteration(A, b)
end_time = time.time()
execution_gauss_time = end_time - start_time
print("Czas wykonania Gauss-Seidel:", execution_gauss_time, "sekund")

# Zadanie C - Rozwiązanie układu równań dla zmienionych wartości a1, a2, a3
a1 = 3
A_new, _ = create_equation_system(N, a1, a2, a3)
x_jacobi_new, residuals_jacobi_new = jacobi_iteration(A_new, b)
x_gauss_seidel_new, residuals_gauss_seidel_new = gauss_seidel_iteration(A_new, b)

# Zadanie D - Rozwiązanie układu równań metodą faktoryzacji LU
x_lu, residual_lu = lu_factorization(A_new, b)

# Zadanie B - Wykres zmiany normy residuum dla metody Jacobiego
plt.plot(residuals_jacobi)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Residual Norm')
plt.title('Convergence of Jacobi Method')
plt.show()

# Zadanie B - Wykres zmiany normy residuum dla metody Gaussa-Seidla
plt.plot(residuals_gauss_seidel)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Residual Norm')
plt.title('Convergence of Gauss-Seidel Method')
plt.show()

# Zadanie B - Wykres zmiany normy residuum dla zmodyfikowanych parametrów
plt.plot(residuals_jacobi, label='Jacobi')
plt.plot(residuals_gauss_seidel, label='Gauss-Seidel')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Residual Norm')
plt.title('Convergence of Methods for Modified Parameters')
plt.legend()
plt.show()

# Zadanie C - Wykres zmiany normy residuum dla zmodyfikowanych parametrów
plt.plot(residuals_jacobi_new, label='Jacobi')
plt.plot(residuals_gauss_seidel_new, label='Gauss-Seidel')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Residual Norm')
plt.title('Convergence of Methods for Modified Parameters')
plt.legend()
plt.show()

# Zadanie D - Wyświetlenie wyniku oraz normy residuum dla metody faktoryzacji LU
print("Solution using LU Factorization:")
print("Solution (x):", x_lu)
print("Residual Norm:", residual_lu)

# Zadanie E - Wykres zależności czasu wyznaczenia rozwiązania dla trzech metod
Ns = [100, 500, 1000, 2000, 3000]
times_jacobi = []
times_gauss_seidel = []
times_lu = []

for N in Ns:
    A, b = create_equation_system(N, a1, a2, a3)

    start_time = time.time()
    jacobi_iteration(A, b)
    end_time = time.time()
    times_jacobi.append(end_time - start_time)

    start_time = time.time()
    gauss_seidel_iteration(A, b)
    end_time = time.time()
    times_gauss_seidel.append(end_time - start_time)

    start_time = time.time()
    lu_factorization(A, b)
    end_time = time.time()
    times_lu.append(end_time - start_time)

plt.plot(Ns, times_jacobi, label='Jacobi')
plt.plot(Ns, times_gauss_seidel, label='Gauss-Seidel')
plt.plot(Ns, times_lu, label='LU Factorization')
plt.xlabel('Size of N')
plt.ylabel('Time (s)')
plt.title('Time vs. Size of N for Different Methods')
plt.legend()
plt.show()
