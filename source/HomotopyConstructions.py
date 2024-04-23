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
            y_prev = sp.lambdify(x, y_prev, 'numpy')( self.x_shot[x] )
            y_next = self.surface.interface_functions[next_interface]
            y_next = sp.lambdify(x, y_next, 'numpy')( self.x_shot[k] )

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


    def _build_bk(self, k):
        pass


    def _build_ck(self, k):
        pass


    def _build_jacobian_matrix_A(self):

        first_derivatives, second_derivatives = self._get_first_second_derivatives(self)
        ak = self._build_ak(self)
        bk = self._build_bk(self)
        ck = self._build_ck(self)

        return np.array([bk, ak, ck])


    def _build_bidiagonal_matrix_B(self):
        pass


    # final function for solving via homotopy
    def solve(self):

        # it runs modifying the attribute self.x_shot according to homotopy
        Axv = self._build_jacobian_matrix_A(self)
        Bxv = self._build_bidiagonal_matrix_B(self)



        pass
