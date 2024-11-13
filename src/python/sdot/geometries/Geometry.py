# from .bindings.loader import sdot_module_for
import numpy as np

class AxisAlignedBox:
    """ space subset, defined by min and max positions
    """

    def __init__( self, min_pos, max_pos ):
        """ 
        """
        self.min_pos = min_pos
        self.max_pos = max_pos

    @staticmethod
    def unit_box( self, ndim ):
        return AxisAlignedBox( np.full( [ ndim ], 0 ), np.full( [ ndim ], 1 ) )

