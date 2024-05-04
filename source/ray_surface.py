# on development
class Ray:

    def __init__(self, trail, interfaces):
        self.segments = len(trail)
        self.trail = trail
        self.interfaces = interfaces
        pass


# on development
class Surface:

    '''
    args:
        interface_functions (list): list of sympy functions representing
            every interface function
        ray (Ray): a created Ray object
        medium_velocities (dict): a dictionary containing a velocity for every
            medium and every case ('P' and 'S')
    '''

    def __init__(self, interface_functions, ray: Ray, medium_velocities: dict):
        self.medium = ['P', 'S']
        self.ray = ray
        self.interface_functions = interface_functions
        self.medium_velocities = medium_velocities
        pass


    def get_cases(self):
        import itertools
        segment_iterator = itertools.product(self.medium,
                                             repeat=self.ray.segments)
        all_cases = [list(segment) for segment in segment_iterator]
        return all_cases


    def get_velocities_vector(self, case: list):
        '''
        args:
            case (list) : is the specific combination of 'P' and 'S' to get all the mapped
                velocities throughout the trail.
        returns:
            velocities (array) : every velocity related to the specific case of 'P' and 'S'.
        '''
        import numpy as np
        
        trail = np.array(self.ray.trail) - 1
        velocities = []

        for medium, trail_num in zip(case, trail):
            velocities += [ self.medium_velocities[medium][trail_num] ]

        return np.array(velocities)
    