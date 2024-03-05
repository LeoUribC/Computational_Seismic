from ray_surface import *


# on development
class Trajectory:

    def __init__(self, ray, surface):
        self.ray = ray
        self.surface = surface
        pass

    def _get_j(self, m, medium):
        return 2*m if medium=='P' else 2*m+1

    def _assign_segment_signature(self, current_m_index, trail_size, ji_pair):
        return ji_pair if current_m_index < trail_size-1 else ji_pair[0]

    def _get_signature(self, cases):
        """
        inputs:
            cases (list): A sequence of 'P' and 'S'
        outputs:
            signature (list): A list of pairs [j, i]
            and one last j for a single sequence of 'P' and 'S'
        """
        trail = self.ray.trail
        interfaces = self.ray.interfaces
        signature = []
        current_m_index = 0
        for medium, t, i in zip(cases, trail, interfaces):
            j = self._get_j(t, medium)
            segment_signature = self._assign_segment_signature(
                current_m_index, self.ray.segments, [j,i]
            )
            signature.append(segment_signature)
            current_m_index += 1
        return signature

    def get_signatures(self):
        all_cases = self.surface.get_cases()
        signatures = []
        for current_case in all_cases:
            case_signature = self._get_signature(current_case)
            signatures.append(case_signature)
        return signatures

    def pretty_print_signatures(self):
        print(f"\nSignatures for trail {self.ray.trail} :\n")
        for case_, signature in zip(self.surface.get_cases(), self.get_signatures()):
            print(f"{case_} = {signature}")
        pass