# on development
class Ray:

    def __init__(self, segments, trail):
        self.segments = segments
        self.trail = trail
        pass


# on development
class Surface:

    def __init__(self, ray):
        self.medium = ['P', 'S']
        self.ray = ray
        pass

    def get_cases(self):
        import itertools
        segment_iterator = itertools.product(self.medium,
                                             repeat=self.ray.segments)
        all_cases = [list(segment) for segment in segment_iterator]
        return all_cases