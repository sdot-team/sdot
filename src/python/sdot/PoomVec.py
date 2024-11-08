from .bindings.loader import poom_vec_module_for, normalized_dtype
import numpy as np

class PoomVec:
    """ Potentially Out of Memory Vector.

        Basically a Wrapper around cpp sdot::PoomVec class 
    
    """

    def __init__( self, values = None, shape = None, dtype = None, _vec = None, _module = None ):
        """ 
        """

        # already a PoomVec ?
        if isinstance( values, PoomVec ):
            self._module = values._module
            self._vec = values._vec
            return
        
        # try to convert values to a numpy array
        try:
            if len( values ):
                array = np.ascontiguousarray( values )
                if array.ndim:
                    if shape is None:
                        shape = array.shape
                    if dtype is None:
                        dtype = array.dtype
                    dtype = normalized_dtype( dtype )

                    self._module = poom_vec_module_for( scalar_type = dtype, item_type = dtype )
                    self._vec = self._module.make_PoomVec_from_ndarray( array )
                    return
        except TypeError:
            pass

        # specification of attriutes
        if _vec is not None:
            self._module = _module
            self._vec = _vec
            return
        
        # default values
        if values is None:
            self._module = None
            self._vec = None
            return
        
        # :P
        raise RuntimeError( f"don't known how to construct a PoomVec using the given arguments" )

    def __repr__( self ):
        return self._vec.__repr__()

    # @property
    # def vertex_coords( self ):
    #     """ get list of coordinates for each vertex (stored in a numpy array) in the current base """
    #     return self._vec.vertex_coords( True )

