import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from matrix_ops import matrix_multiply, matrix_transpose, matrix_determinant, matrix_inverse, lu_decomposition, is_singular, plot_2d_transformation

def test_matrix_multiply():
    A = [[1, 2], [3, 4]]
    B = [[2, 0], [1, 2]]
    assert matrix_multiply(A, B) == [[4, 4], [10, 8]]

def test_determinant():
    A = [[1, 2], [3, 4]]
    assert np.isclose(matrix_determinant(A), -2)

def test_is_singular():
    A = [[1, 2], [2, 4]]
    assert is_singular(A)
    
def test_inverse():
    A = [[4, 7], [2, 6]]
    inv = matrix_inverse(A)
    expected = np.linalg.inv(A)
    assert np.allclose(inv, expected)
    
def test_lu():
    A = [[4, 3], [6, 3]]
    L, U = lu_decomposition(A)
    assert np.allclose(matrix_multiply(L, U), A)
