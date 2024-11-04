from .bindings.loader import module_for
import numpy as np

class Cell:
    """ wrapper around cpp sdot::Cell class """

    def __init__( self, ndim = None, dtype = None, _cell = None ):
        """ _cell is the sdot::Cell class instance. If not specified, __init__ makes a new one using `ndim` and `dtype`.
            By default, dtype = np.double, ndim = 2
        """

        if _cell:
            self._binding_module = _cell._binding_module
            self._cell = _cell
            return

        if ndim is None:
            ndim = 2

        if dtype is None:
            dtype = "FP64"

        self._binding_module = module_for( scalar_type = dtype, nb_dims = ndim )
        self._cell = self._binding_module.Cell()

    def __repr__( self ):
        return self._cell.__repr__()

    def vertex_coords( self, allow_lower_dim = False ):
        """ get list of coordinates for each vertex (stored in a numpy array) 
        
            if allow_lower_dim == True, it will return the data for the cell with dim == true_dimensionality (projected on self.base())
        """
        return self._cell.vertex_coords( allow_lower_dim )

    def vertex_refs( self, allow_lower_dim = False ):
        """ get list of cut indices for each vertex (stored in a numpy array) 
        
            if allow_lower_dim == True, it will return the data for the cell with dim == true_dimensionality (projected on self.base())
        """
        return self._cell.vertex_refs( allow_lower_dim )

    @property
    def true_dimensionality( self ):
        """ at the beginning, all the cell starts with true_dimensionality == 0. A first cut will lead to true_dimensionality == 1. 
            If not in the same direction, true_dimensionality will be incremented, ...
            If true_dimensionality < ndims, coordinates are expressed in self.base
        """
        return self._cell.true_dimensionality

    @property
    def nb_active_cuts( self ):
        """ beware: if remove_inactive_cuts() has not been called, this procedure will have first to compute the active cuts (may take some time) """
        return self._cell.nb_active_cuts

    @property
    def nb_stored_cuts( self ):
        """ may be different from nb_active_cuts if remove_inactive_cuts() has not been called """
        return self._cell.nb_active_cuts

    @property
    def nb_vertices( self ):
        return self._cell.nb_vertices

    @property
    def bounded( self ):
        return self._cell.bounded

    @property
    def empty( self ):
        return self._cell.empty
    
    def cut( self, dir, val, index = 0 ):
        return self._cell.cut( np.ascontiguousarray( dir ), val, int( index ) )

    def display_vtk( self, vtk_output ):
        return self._cell.display_vtk( vtk_output )
