import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B: raise ValueError("Invalid dimensions")
    C = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return C

def matrix_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matrix_determinant(A):
    n = len(A)
    if n == 1: return A[0][0]
    if n == 2: return A[0][0]*A[1][1] - A[0][1]*A[1][0]
    det = 0
    for c in range(n):
        sub = [[A[i][j] for j in range(n) if j != c] for i in range(1, n)]
        det += ((-1)**c) * A[0][c] * matrix_determinant(sub)
    return det

def _get_minor(A, i, j):
    return [[A[r][c] for c in range(len(A)) if c != j] for r in range(len(A)) if r != i]

def matrix_inverse(A):
    det = matrix_determinant(A)
    if det == 0: raise ValueError("Singular matrix")
    n = len(A)
    if n == 1: return [[1/det]]
    cofactors = []
    for r in range(n):
        cofactor_row = []
        for c in range(n):
            minor = _get_minor(A, r, c)
            cofactor_row.append(((-1)**(r+c)) * matrix_determinant(minor))
        cofactors.append(cofactor_row)
    adjugate = matrix_transpose(cofactors)
    return [[adjugate[r][c]/det for c in range(n)] for r in range(n)]

def lu_decomposition(A):
    n = len(A)
    L = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    U = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if U[i][i] == 0: continue
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            for k in range(i, n):
                U[j][k] -= factor * U[i][k]
    return L, U

def is_singular(A):
    try:
        return abs(matrix_determinant(A)) < 1e-9
    except:
        return True

def plot_2d_transformation(matrix, points):
    transformed = matrix_multiply(matrix, points)
    plt.figure()
    plt.scatter(points[0], points[1], c='b', label='Original')
    plt.scatter(transformed[0], transformed[1], c='r', label='Transformed')
    plt.legend()
    plt.savefig('transform.png')
    plt.close()
    return transformed
