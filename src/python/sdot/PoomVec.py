from .bindings.loader import poom_vec_module_for
import numpy as np

class PoomVec:
    """ Potentially Out of Memory Vector.

        Basically a Wrapper around cpp sdot::PoomVec class 
    
    """

    def __init__( self, values = None, shape = None, dtype = None, _vec = None, _module = None ):
        """ 
        """

        if isinstance( values, PoomVec ):
            self._module = values._module
            self._vec = values._vec
        elif isinstance( values, np.ndarray ):
            array = np.ascontiguousarray( values )
            if shape is None:
                shape = array.shape
            if dtype is None:
                dtype = array.dtype

            self._module = poom_vec_module_for( scalar_type = dtype, shape = shape )
            self._vec = self._module.make_PoomVec_from_ndarray( array )
        elif _vec is not None:
            self._module = _module
            self._vec = _vec
        elif values is None:
            self._module = None
            self._vec = None
        else:
            raise RuntimeError( f"don't known how to construct a PoomVec using the given arguments" )

    def __repr__( self ):
        return self._vec.__repr__()

    # @property
    # def vertex_coords( self ):
    #     """ get list of coordinates for each vertex (stored in a numpy array) in the current base """
    #     return self._vec.vertex_coords( True )

