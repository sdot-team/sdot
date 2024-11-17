# from .Cell import Cell
import numpy as np

class PowerDiagramSummary:
    """
        Aggregation of arrays that represents a fully computed PowerDiagram.

        In these arrays, if < `nb_cells`, a `ref` represents a cell index. Else (if >= `nb_cells`) it represents a boundary index + `nb_cells` (i.e. `boundary_index = ref - summary.nb_cells`).
        
        Attributes:
            `vertex_coords`: coordinates for each vertex
            `ref_lists`: `ref_lists[ d ]` gives the list of refs for each item of dimensionality `d`. For instance, `ref_lists[ 0 ]` gives the refs (cells of boundary indices) for each vertex, `ref_lists[ 1 ]` gives the refs for each edge, ... It describes how items has been made (from which cell and boundary intersections)
            `parenting`: `parenting[ r ][ c ]` gives parenting (parents, children or siblings depending or `r - c`) of items of dimensionality `c` for  items of dimensionality `r`
            `base`: can be use if `true_dimensionality` < `ndim` to get coordinate in the `ndim` space

    """
    def __init__( self,vertex_coords, ref_lists, parenting, boundary_items, base ):
        self.boundary_items = boundary_items
        self.vertex_coords  = vertex_coords
        self.ref_lists      = ref_lists
        self.parenting      = parenting
        self.base           = base

    @property
    def nb_cells( self ):
        return len( self.vertex_coords )

    @property
    def true_dimensionality( self ):
        return self.base.shape[ 1 ]

    @property
    def ndim( self ):
        return self.base.shape[ 0 ]

    def nb_items( self, dim ):
        """ nb items of dimensionality `dim` """
        return len( self.parenting[ dim ][ 0 ] )

    def barycenters( self, dim = None ):
        """ by default, return the barycenters of the cells.

            specifiying dim allows for returning the barycenters of vertices (dim=0), edges (dim=1), and so on.

            TODO: underlying_measures
        """
        if dim is None:
            dim = self.ndim

        if dim == 0:
            return self.vertex_coords

        res = np.empty( [ self.nb_items( dim ), self.ndim ] )
        for n, vs in enumerate( self.parenting[ dim ][ 0 ] ):
            res[ n, : ] = np.mean( self.vertex_coords[ vs ], axis=0 )
 
        return res
 
    def integrals( self ):
        """  """
        raise RuntimeError( "TODO" )
    
    