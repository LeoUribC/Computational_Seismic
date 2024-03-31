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


def get_jacobian(F, vector, h=1e-5):
    """
    Calcula aproximacion para la matriz Jacobiana
    evaluada en un vector
    """

    h_for, h_back = build_matrices(vector, h)
    J = ( F(h_for) - F(h_back) ) / (2*h)

    return J


def get_k_values(F, wj, X0, h):
    """
    Proceso para obtener los vectores kn
    """

    b = -h*F(X0)

    k1 = np.linalg.solve( get_jacobian(F, wj), b )
    k2 = np.linalg.solve( get_jacobian(F, wj + 0.5*k1), b )
    k3 = np.linalg.solve( get_jacobian(F, wj + 0.5*k2), b )
    k4 = np.linalg.solve( get_jacobian(F, wj + k3), b )

    return k1, k2, k3, k4


def get_solution(F, X0, h):
    
    lambdas = np.arange(0.0, 1.0, h)

    k1, k2, k3, k4 = get_k_values(F, X0, X0, h)
    w = X0 + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    for _ in lambdas[1:]:
        k1, k2, k3, k4 = get_k_values(F, w, X0, h)
        w = w + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return w
