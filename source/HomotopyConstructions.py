"""
Calculations to homotopy, continuing on velocities, receivers and
finally iterate with Newton Method to get final x positions of rays
"""

from ray_surface import *
from ray_track import *
import numpy as np
import sympy as sp
from DefinedErrors import *



class SystemBuilder:


    # surface contains first shot and vector of velocities
    # first shot is named X0 and vector of velocities is V

    def __init__(self, x_shot, surface: Surface):
        self.surface = surface  # a Surface object
        self.ray = self.surface.ray  # a ray object
        self.x_shot = x_shot
        pass



    # keep in mind that surface is going to store the equations as sympy expressions
    def _get_first_second_derivatives(self):
        import sympy as sp

        x = sp.symbols('x')
        first_derivatives = [ interface.diff(x, 1) for interface in self.surface.interface_functions ]
        second_derivatives = [ interface.diff(x, 2) for interface in self.surface.interface_functions ]
        
        return first_derivatives, second_derivatives



    def _build_ak(self, X, next_case):

        x, k = sp.symbols('x'), 1
        N, ak = len(X)-2, []

        first_derivatives, second_derivatives = self._get_first_second_derivatives()

        # update for every guess

        while k <= N:

            curr_interface = self.ray.interfaces[k]
            prev_interface = self.ray.interfaces[k-1]
            next_interface = self.ray.interfaces[k+1]

            y_curr = self.surface.interface_functions[curr_interface]
            y_curr = sp.lambdify(x, y_curr, 'numpy')( X[k] )
            y_prev = self.surface.interface_functions[prev_interface]
            y_prev = sp.lambdify(x, y_prev, 'numpy')( X[k-1] )
            y_next = self.surface.interface_functions[next_interface]
            y_next = sp.lambdify(x, y_next, 'numpy')( X[k+1] )

            yprime_curr = first_derivatives[curr_interface]
            yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( X[k] )
            ypprime_curr = second_derivatives[curr_interface]
            ypprime_curr = sp.lambdify(x, ypprime_curr, 'numpy')(X[k])

            Vk_curr = self.surface.get_velocities_vector(next_case)[k-1]
            Vk_next = self.surface.get_velocities_vector(next_case)[k]

            dy_curr = y_curr - y_prev
            dy_next = y_next - y_curr
            dx_curr = X[k] - X[k-1]
            dx_next = X[k+1] - X[k]

            D_curr = ( dx_curr**2 + dy_curr**2 )**0.5
            D_next = ( dx_next**2 + dy_next**2 )**0.5

            ak += [ (Vk_next/D_curr) * ( 1 + ypprime_curr * dy_curr + yprime_curr**2 -\
                                         ( (dx_curr + yprime_curr*dy_curr)/D_curr )**2 ) +\
                    (Vk_curr/D_next) * ( 1 - ypprime_curr * dy_next + yprime_curr**2 -\
                                         ( (dx_next + yprime_curr*dy_next)/D_next )**2 ) ]
            k += 1

        return np.array(ak)



    def _build_bk(self, X, next_case):

        x, k = sp.symbols('x'), 2
        N, bk = len(X)-2, []

        first_derivatives, _ = self._get_first_second_derivatives()

        # update every self.x_shot

        while k <= N:

            curr_interface = self.ray.interfaces[k]
            prev_interface = self.ray.interfaces[k-1]

            y_curr = self.surface.interface_functions[curr_interface]
            y_curr = sp.lambdify(x, y_curr, 'numpy')( X[k] )
            y_prev = self.surface.interface_functions[prev_interface]
            y_prev = sp.lambdify(x, y_prev, 'numpy')( X[k-1] )

            yprime_curr = first_derivatives[curr_interface]
            yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( X[k] )
            yprime_prev = first_derivatives[prev_interface]
            yprime_prev = sp.lambdify(x, yprime_prev, 'numpy')( X[k-1] )

            Vk_next = self.surface.get_velocities_vector(next_case)[k]

            dy_curr = y_curr - y_prev
            dx_curr = X[k] - X[k-1]

            D_curr = ( dx_curr**2 + dy_curr**2 )**0.5

            bk += [ -(Vk_next/D_curr) * ( 1 + yprime_prev*yprime_curr -\
                    ( (dx_curr + yprime_prev*dy_curr)/D_curr ) * ( (dx_curr + yprime_curr*dy_curr)/D_curr ) ) ]
            k += 1

        return np.array(bk)



    def _build_ck(self, X, next_case):

        x, k = sp.symbols('x'), 1
        N1, ck = len(X)-3, []

        first_derivatives, _ = self._get_first_second_derivatives()

        # update every self.x_shot

        while k <= N1:

            curr_interface = self.ray.interfaces[k]
            next_interface = self.ray.interfaces[k+1]

            y_curr = self.surface.interface_functions[curr_interface]
            y_curr = sp.lambdify(x, y_curr, 'numpy')( X[k] )
            y_next = self.surface.interface_functions[next_interface]
            y_next = sp.lambdify(x, y_next, 'numpy')( X[k+1] )

            yprime_curr = first_derivatives[curr_interface]
            yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( X[k] )
            yprime_next = first_derivatives[next_interface]
            yprime_next = sp.lambdify(x, yprime_next, 'numpy')( X[k+1] )

            Vk_curr = self.surface.get_velocities_vector(next_case)[k-1]

            dy_next = y_next - y_curr
            dx_next = X[k+1] - X[k]

            D_next = ( dx_next**2 + dy_next**2 )**0.5

            ck += [ -(Vk_curr/D_next) * ( 1 + yprime_curr*yprime_next -\
                    ( (dx_next + yprime_curr*dy_next)/D_next ) * ( (dx_next + yprime_next*dy_next)/D_next ) ) ]
            k += 1

        return np.array(ck)


    def _build_jacobian_matrix_A(self, X, next_case):

        ak = self._build_ak(X, next_case)
        bk = self._build_bk(X, next_case)
        ck = self._build_ck(X, next_case)

        # creating diagonals
        ak_diag = np.diag(ak)
        bk_diag = np.diag(bk, -1)
        ck_diag = np.diag(ck, 1)

        return ak_diag + bk_diag + ck_diag  # tridiagonal matrix


    # function to get a bidiagonal matrix
    def _build_bidiagonal_matrix_B(self, X):

        N, k = len(X)-2, 1   # fix dimensions
        x = sp.symbols('x')
        first_derivatives, _ = self._get_first_second_derivatives()

        B = np.zeros( (N, N+1) )
        main_diag, upper_diag = [], []
        
        while k <= N:

            curr_interface = self.ray.interfaces[k]
            prev_interface = self.ray.interfaces[k-1]
            next_interface = self.ray.interfaces[k+1]

            y_prev = self.surface.interface_functions[prev_interface]
            y_prev = sp.lambdify(x, y_prev, 'numpy')( X[k-1] )
            y_curr = self.surface.interface_functions[curr_interface]
            y_curr = sp.lambdify(x, y_curr, 'numpy')( X[k] )
            y_next = self.surface.interface_functions[next_interface]
            y_next = sp.lambdify(x, y_next, 'numpy')( X[k+1] )

            yprime_curr = first_derivatives[curr_interface]
            yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( X[k] )

            dy_curr = y_curr - y_prev
            dy_next = y_next - y_curr
            dx_curr = X[k] - X[k-1]
            dx_next = X[k+1] - X[k]

            D_curr = ( dx_curr**2 + dy_curr**2 )**0.5
            D_next = ( dx_next**2 + dy_next**2 )**0.5

            main_diag += [ -( (dx_next + yprime_curr * dy_next)/D_next ) ]
            upper_diag += [ ( dx_curr + yprime_curr * dy_curr )/D_curr ]
            
            k += 1
        
        # building B
        diag = np.arange(N)
        B[diag, diag] = main_diag
        B[diag, diag + 1] = upper_diag

        return B
    


    def _build_receiver_matrix_B(self, X):
        pass



    # function for solving via homotopy, continuation is made on velocity
    def get_first_ray(self, X, data_cases, continuation='v', dl=0.25):
        '''
        args:
            * X (1D-array): Array containing every point in X to be solved.
            * data_cases (2D-array): A 2D array containing 2 cases to calculate\
            both velocities for known and desired case. Every case must be a\
            sequence of 'P' and 'S'.
        returns:
            * final_guess (1D-array): a good first candidate of aproximations to\
            be passed to newton method.
        '''

        iterations = np.arange(0, 1+dl, dl)
        final_guess = X

        curr_case, next_case = data_cases

        # PENDING: implement function to get receiver vector based on ray trail
        continuation_options = {'v': [ self.surface.get_velocities_vector(curr_case),
                                       self.surface.get_velocities_vector(next_case) ],
                                'r': [ ]}  # implement later, a method called 'self.surface.get_receiver_vector()'
        
        curr_V, next_V = continuation_options[continuation]

        # it runs modifying the attribute self.x_shot according to homotopy, iteration goes here
        for _ in iterations:

            Axv = self._build_jacobian_matrix_A(final_guess, next_case)
            Bxv = self._build_bidiagonal_matrix_B(final_guess)

            x_dot = np.linalg.solve( Axv, -np.dot(Bxv, next_V - curr_V) )
            final_guess = final_guess + dl*x_dot

        return final_guess



    def _snell_law_equation(self, k, X):
        '''
        docstring
        X must be every updated X vector from newton
        '''

        x = sp.symbols('x')

        # (pending: change the velocities' source, surface velocities not follow signature velocities)
        v_curr, v_next = self.surface.V[k], self.surface.V[k+1]
        x_prev, x_curr, x_next = X[k-1], X[k], X[k+1]

        curr_interface = self.ray.interfaces[k]
        prev_interface = self.ray.interfaces[k-1]
        next_interface = self.ray.interfaces[k+1]

        y_prev = self.surface.interface_functions[prev_interface]
        y_prev = sp.lambdify(x, y_prev, 'numpy')( x_prev )
        y_curr = self.surface.interface_functions[curr_interface]
        y_curr = sp.lambdify(x, y_curr, 'numpy')( x_curr )
        y_next = self.surface.interface_functions[next_interface]
        y_next = sp.lambdify(x, y_next, 'numpy')( x_next )

        first_derivatives, _ = self._get_first_second_derivatives()
        yprime_curr = first_derivatives[curr_interface]
        yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( x_curr )

        ray_before = v_next * ( (x_curr - x_prev) + yprime_curr * ( y_curr - y_prev ) ) /\
            ( (x_curr - x_prev)**2 + (y_curr - y_prev)**2 )**0.5

        ray_after = v_curr * ( (x_next - x_curr) + yprime_curr * (y_next - y_curr) ) /\
            ( (x_next - x_curr)**2 + (y_next - y_curr)**2 )**0.5

        return ray_before - ray_after



    # iteratively calls _snell_law_equation
    def _build_phi_array(self, X):
        '''
        '''
        N = len(self.x_shot)-1
        phi = [ self._snell_law_equation(k_, X) for k_ in range(1, N) ]

        return np.array(phi)



    def newton_solve(self, data_cases, tol=1e-3, continuation='v'):
        '''
        args:
            * data_cases (2D-array) : List of 'P' and 'S' related to known first ray.
            * tol (float) : Error for calculations.
            * continuation (string) : Must be either 'v' for applying continuation on\
            velocities or 'r' for continuation on receivers.
        returns:
            * ray_points (array) : solution of values on x where the ray interacts with\
            each interface throughout its trail.
        '''

        # raises an error for any string other than 'v' or 'r' on continuation parameter
        if continuation not in ['v', 'r']:
            raise ContinuationNotDefinedException
        
        curr_case, next_case = data_cases

        current_X = self.get_first_ray(self.x_shot, data_cases)

        A_newton = self._build_jacobian_matrix_A(current_X, next_case)
        phi = self._build_phi_array(current_X)
        dxv = np.linalg.solve( A_newton, -phi )
        X_sol = current_X + dxv

        while np.max( np.abs(X_sol - current_X) ) > tol:
            current_X = X_sol
            A_newton = self._build_jacobian_matrix_A(current_X, next_case)
            phi = self._build_phi_array(current_X)
            dxv = np.linalg.solve( A_newton, -phi )
            X_sol = current_X + dxv

        return X_sol
