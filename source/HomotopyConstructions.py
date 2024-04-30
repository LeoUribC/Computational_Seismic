"""
La idea es que resuelva para un solo rayo, luego se replica para
todas las posibles combinaciones de rayos
"""

from ray_surface import *
from ray_track import *
import numpy as np
import sympy as sp



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



    def _build_ak(self):

        x, k = sp.symbols('x'), 1
        ak = []

        first_derivatives, second_derivatives = self._get_first_second_derivatives(self)

        # update every self.x_shot

        while k < len(self.x_shot) - 1:

            curr_interface = self.ray.interfaces[k]
            prev_interface = self.ray.interfaces[k-1]
            next_interface = self.ray.interfaces[k+1]

            y_curr = self.surface.interface_functions[curr_interface]
            y_curr = sp.lambdify(x, y_curr, 'numpy')( self.x_shot[k] )
            y_prev = self.surface.interface_functions[prev_interface]
            y_prev = sp.lambdify(x, y_prev, 'numpy')( self.x_shot[k-1] )
            y_next = self.surface.interface_functions[next_interface]
            y_next = sp.lambdify(x, y_next, 'numpy')( self.x_shot[k+1] )

            yprime_curr = first_derivatives[curr_interface]
            yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( self.x_shot[k] )
            ypprime_curr = second_derivatives[curr_interface]
            ypprime_curr = sp.lambdify(x, ypprime_curr, 'numpy')(self.x_shot[k])

            Vk_curr = self.surface.V[k]
            Vk_next = self.surface.V[k+1]

            dy_curr = y_curr - y_prev
            dy_next = y_next - y_curr
            dx_curr = self.x_shot[k] - self.x_shot[k-1]
            dx_next = self.x_shot[k+1] - self.x_shot[k]

            D_curr = ( dx_curr**2 + dy_curr**2 )**0.5
            D_next = ( dx_next**2 + dy_next**2 )**0.5

            ak += [ (Vk_next/D_curr) * ( 1 + ypprime_curr * dy_curr + yprime_curr**2 -\
                                         ( (dx_curr + yprime_curr*dy_curr)/D_curr )**2 ) +\
                    (Vk_curr/D_next) * ( 1 - ypprime_curr * dy_next + yprime_curr**2 -\
                                         ( (dx_next + yprime_curr*dy_next)/D_next )**2 ) ]
            k += 1

        return np.array(ak)



    def _build_bk(self):

        x, k = sp.symbols('x'), 2
        bk = []

        first_derivatives, _ = self._get_first_second_derivatives(self)

        # update every self.x_shot

        while k < len(self.x_shot) - 1:

            curr_interface = self.ray.interfaces[k]
            prev_interface = self.ray.interfaces[k-1]

            y_curr = self.surface.interface_functions[curr_interface]
            y_curr = sp.lambdify(x, y_curr, 'numpy')( self.x_shot[k] )
            y_prev = self.surface.interface_functions[prev_interface]
            y_prev = sp.lambdify(x, y_prev, 'numpy')( self.x_shot[k-1] )

            yprime_curr = first_derivatives[curr_interface]
            yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( self.x_shot[k] )
            yprime_prev = first_derivatives[prev_interface]
            yprime_prev = sp.lambdify(x, yprime_prev, 'numpy')( self.x_shot[k-1] )

            Vk_next = self.surface.V[k+1]

            dy_curr = y_curr - y_prev
            dx_curr = self.x_shot[k] - self.x_shot[k-1]

            D_curr = ( dx_curr**2 + dy_curr**2 )**0.5

            bk += [ -(Vk_next/D_curr) * ( 1 + yprime_prev*yprime_curr -\
                    ( (dx_curr + yprime_prev*dy_curr)/D_curr ) * ( (dx_curr + yprime_curr*dy_curr)/D_curr ) ) ]
            k += 1

        return np.array(bk)



    def _build_ck(self):

        x, k = sp.symbols('x'), 1
        ck = []

        first_derivatives, _ = self._get_first_second_derivatives(self)

        # update every self.x_shot

        while k < len(self.x_shot) - 2:

            curr_interface = self.ray.interfaces[k]
            next_interface = self.ray.interfaces[k+1]

            y_curr = self.surface.interface_functions[curr_interface]
            y_curr = sp.lambdify(x, y_curr, 'numpy')( self.x_shot[k] )
            y_next = self.surface.interface_functions[next_interface]
            y_next = sp.lambdify(x, y_next, 'numpy')( self.x_shot[k+1] )

            yprime_curr = first_derivatives[curr_interface]
            yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( self.x_shot[k] )
            yprime_next = first_derivatives[next_interface]
            yprime_next = sp.lambdify(x, yprime_next, 'numpy')( self.x_shot[k+1] )

            Vk_curr = self.surface.V[k]

            dy_next = y_next - y_curr
            dx_next = self.x_shot[k+1] - self.x_shot[k]

            D_next = ( dx_next**2 + dy_next**2 )**0.5

            ck += [ -(Vk_curr/D_next) * ( 1 + yprime_curr*yprime_next -\
                    ( (dx_next + yprime_curr*dy_next)/D_next ) * ( (dx_next + yprime_next*dy_next)/D_next ) ) ]
            k += 1

        return np.array(ck)


    def _build_jacobian_matrix_A(self):

        ak = self._build_ak(self)
        bk = self._build_bk(self)
        ck = self._build_ck(self)

        # creating diagonals
        ak_diag = np.diag(ak)
        bk_diag = np.diag(bk, -1)
        ck_diag = np.diag(ck, 1)

        return ak_diag + bk_diag + ck_diag  # tridiagonal matrix


    # function to get a bidiagonal matrix
    def _build_bidiagonal_matrix_B(self):

        N, k = len(self.x_shot) - 2, 1
        x = sp.symbols('x')
        first_derivatives, _ = self._get_first_second_derivatives(self)

        B = np.zeros( (N, N+1) )
        main_diag, upper_diag = [], []
        
        while k <= N:

            curr_interface = self.ray.interfaces[k]
            prev_interface = self.ray.interfaces[k-1]
            next_interface = self.ray.interfaces[k+1]

            y_prev = self.surface.interface_functions[prev_interface]
            y_prev = sp.lambdify(x, y_prev, 'numpy')( self.x_shot[k-1] )
            y_curr = self.surface.interface_functions[curr_interface]
            y_curr = sp.lambdify(x, y_curr, 'numpy')( self.x_shot[k] )
            y_next = self.surface.interface_functions[next_interface]
            y_next = sp.lambdify(x, y_next, 'numpy')( self.x_shot[k+1] )

            yprime_curr = first_derivatives[curr_interface]
            yprime_curr = sp.lambdify(x, yprime_curr, 'numpy')( self.x_shot[k] )

            dy_curr = y_curr - y_prev
            dy_next = y_next - y_curr
            dx_curr = self.x_shot[k] - self.x_shot[k-1]
            dx_next = self.x_shot[k+1] - self.x_shot[k]

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
    

    def _get_v_hat(self):
        pass


    # final function for solving via homotopy
    def solve(self, dl):

        iterations = np.arange(0, 1+dl, dl)

        # it runs modifying the attribute self.x_shot according to homotopy, iteration goes here
        for _ in iterations:

            Axv = self._build_jacobian_matrix_A(self)
            Bxv = self._build_bidiagonal_matrix_B(self)
            v_hat = self._get_v_hat(self)
            x_dot = np.linalg.solve( Axv, -np.dot(Bxv, self.surface.V - v_hat) )
            self.x_shot = self.x_shot + dl*x_dot

        return self.x_shot
