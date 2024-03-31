"""
This module implements homotopy and continuation
method using runge-kutta method.
"""

import numpy as np


def build_matrices(vector, h):

    n = len(vector)
    main_matrix = np.repeat(vector, n).reshape(n, n)

    forward  = main_matrix.copy()
    backward = main_matrix.copy()

    np.fill_diagonal( forward,
                      forward.diagonal() + h )
    np.fill_diagonal( backward,
                      backward.diagonal() - h )
    
    return forward, backward


def get_jacobian(F, vector, h=1e-3):
    """
    Calcula aproximacion para la matriz Jacobiana
    evaluada en un vector
    """

    h_for, h_back = build_matrices(vector, h)
    J = ( F(h_for) - F(h_back) ) / (2*h)

    return J


def get_k_values():
    """
    Proceso para obtener los vectores kn
    """
    
    k1, k2, k3, k4, wj = 0
    return k1, k2, k3, k4, wj


def get_next_xlambda():
    pass


def get_solution(X0, h=1e-2):
    
    lambdas = np.arange(0, 1+h, h)
    xapprox = X0

    while True:
        xapprox = get_next_xlambda(xapprox)
    
    return xapprox