import numpy as np

def gauss_jordan_elimination(matrix):
    rows, cols = len(matrix), len(matrix[0])

    for i in range(rows):
        pivot = matrix[i][i]
        for j in range(cols):
            matrix[i][j] /= pivot

        for k in range(rows):
            if k != i:
                factor = matrix[k][i]
                for j in range(cols):
                    matrix[k][j] -= factor * matrix[i][j]

    return matrix

def jacobi_iteration(A, b, x0, max_iterations=50, tolerance=1e-6):
    n = len(A)
    x = np.copy(x0)

    for iteration in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :n], x_old) - A[i, i] * x_old[i]
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            break

    return x

def gauss_seidel_iteration(A, b, x0, max_iterations=50, tolerance=1e-6):
    n = len(A)
    x = np.copy(x0)

    for iteration in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            break

    return x

# Contoh penggunaan
if __name__ == "__main__":
    # Masukkan matriks koefisien A dan vektor hasil b
    A = np.array([[2, -1, 1],
                  [3, 3, 9],
                  [1, 1, 1]], dtype=float)

    b = np.array([8, 24, 6], dtype=float)

    # Inisialisasi solusi awal
    x0 = np.zeros_like(b, dtype=float)

    # Metode Gauss-Jordan
    augmented_matrix = np.column_stack((A, b))
    gauss_jordan_result = gauss_jordan_elimination(augmented_matrix)

    print("Gauss-Jordan Elimination:")
    print(gauss_jordan_result[:, -1])

    # Metode Jacobi
    jacobi_result = jacobi_iteration(A, b, x0)

    print("\nJacobi Iteration:")
    print(jacobi_result)

    # Metode Gauss-Seidel
    gauss_seidel_result = gauss_seidel_iteration(A, b, x0)

    print("\nGauss-Seidel Iteration:")
    print(gauss_seidel_result)
