"""The seeds at the centres of the cells of a regular grid ( `SumOfDiracsOnAGrid`, `distributions/SumOfDiracsOnAGrid.Grid` ):
no positions are stored, only the grid -- `grid_origin`, `grid_frame` ( the edges of one cell, its rows ) and `grid_shape`.
Seed `k` is the cell `k` in C order, at `grid_origin + grid_frame^T ( i + 1/2 )`.

`PowerDiagram( None, weights, accelerator = grid )`. Each cell is cut by the ring of its adjacent seeds, then by what an
implicit tree over the grid cannot rule out ( `PowerDiagram_Grid.h::ProviderGrid` ), whose nodes carry the affine majorants of
the weights ( `node_wa`, `node_wb` ), redone by one kernel when the weights change. Kernel side: `PowerDiagram_Grid.h`.
"""

import numpy as np

import loom
from loom.compilation.FfiCode import FfiCode
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, new_batch_axis
from loom.util import Aggregate

from .PowerDiagram import PowerDiagram


class PowerDiagram_Grid( PowerDiagram ):
    grid_origin : RealTensor[ "dim" ]
    grid_frame  : RealTensor[ "dir", "dim" ]
    grid_shape  : IntTensor[ "dim", dict( size = 32 ) ]

    # OPTIONAL, as for `PowerDiagram_Plain`: absent, the Euclidean diagram ( and no majorants )
    weights     : RealTensor[ "num_point" ]

    # the affine majorants of the weights, per node of the implicit tree ( `GridLevels` in `PowerDiagram_Grid.h` )
    node_wa     : RealTensor[ "num_gnode", "dim" ]
    node_wb     : RealTensor[ "num_gnode" ]
    num_gnode   : Axis[ "nb_gnodes" ]
    nb_gnodes   : ShapeVar

    def _init_seeds( self, positions, weights, accelerator ):
        g = accelerator
        d = len( g.shape )
        res = dict( grid_origin = np.asarray( g.origin, dtype = float ).reshape( d ),
                    grid_frame = np.asarray( g.cell_frame, dtype = float ).reshape( d, d ),
                    grid_shape = np.asarray( g.shape, dtype = np.int32 ),
                    nb_gnodes = grid_nb_nodes( g.shape ) )
        if weights is not None:
            res[ "weights" ] = weights
        return res

    def __init__( self, positions, weights = None, *args, accelerator = None, **kwargs ):
        super().__init__( positions, weights, *args, accelerator = accelerator, **kwargs )
        #: the grid of the seeds ( `SumOfDiracsOnAGrid.Grid` )
        self.grid = accelerator
        if self.weights.is_defined:
            self._refresh_majorants()

    @property
    def positions( self ):
        """`[ n, d ]`, COMPUTED from the grid ( only for what asks for them: the cells do not )"""
        from .distributions.SumOfDiracsOnAGrid import _centres
        g = self.grid
        return RealTensor[ self.num_point, self.dim ]( _centres( g.shape ) @ g.cell_frame + g.origin[ None, : ] )

    @positions.setter
    def positions( self, positions ):
        raise NotImplementedError( "the seeds of a grid do not move: build a `PowerDiagram` on positions" )

    def __setattr__( self, name, value ):
        # new weights: the seeds do not move, only the majorants of the nodes are redone ( one kernel over the nodes )
        super().__setattr__( name, value )
        if name == "weights" and not self.__dict__.get( "_quiet_weights" ) and self.__dict__.get( "grid" ) is not None \
                and self.weights.is_defined:
            self._refresh_majorants()

    def _ranks_of_items( self ):
        return np.arange( int( self.nb_points.value ) )

    def _grad_seeds_expr( self ):
        # ( the seeds do not move: no cotangent of the positions, those of the weights )
        return "sdot::NoneTensor<double,sdot::Tuple<SI,SI>,sdot::Tuple<>>(), grad_of_inputs.power_diagram.weights"

    def ffi_units( self, call_arg ):
        return []                                        # ( no cell engine unit: the generic walk with `ProviderGrid` )

    def _refresh_majorants( self ):
        """the affine majorant of the weights of each node of the implicit tree, one work-item per node"""
        num_node = new_batch_axis( int( self.nb_gnodes.value ), prefix = "gridnode" )
        majorant = _GridMajorant( nb_dims = int( self.nb_dims.value ), batch_axes = [ num_node ] )
        loom.ffi_call(
            "grid_refresh_majorants",
            FfiCode.per_item( includes = [ "sdot/PowerDiagram_Grid.h" ],
                code = "inputs.power_diagram.node_majorant( SI( batch_index.values[ 0 ].index ), outputs.majorant.wa( batch_index ), "
                       "outputs.majorant.wb( batch_index ) );" ),
            power_diagram = self,
            majorant = loom.out( majorant ),
            has_dynamic_capacity = False,
        )
        self.node_wa = majorant.wa.raw
        self.node_wb = majorant.wb.raw

    # ---- what the `SdotPlanNd` solver writes ( see `PowerDiagram_Bsp` ) ----------------------------

    def _solver_weights_call( self ):
        w = RealTensor[ self.num_point ]()
        wa = RealTensor[ self.num_gnode, self.dim ]()
        wb = RealTensor[ self.num_gnode ]()
        args = dict( weights_out = w, node_wa_out = wa, node_wb_out = wb )
        return ( "inputs.power_diagram.with_weights( outputs.weights_out, outputs.node_wa_out, outputs.node_wb_out )",
                 { n: loom.out( t ) for n, t in args.items() }, args )

    def _solver_weights_after( self, produced ):
        self._quiet_weights = True                       # ( the solver wrote the majorants of these weights )
        self.weights = produced[ "weights_out" ].raw
        self._quiet_weights = False
        self.node_wa = produced[ "node_wa_out" ]
        self.node_wb = produced[ "node_wb_out" ]


def grid_nb_nodes( shape ):
    """the nodes of the implicit tree over a grid of `shape` ( `GridLevels` in `PowerDiagram_Grid.h`: same layout )"""
    d = len( shape )
    leaf = 4 if d == 2 else 2
    nb, l = 0, 0
    while True:
        side = leaf << l
        count = [ ( int( s ) + side - 1 ) // side for s in shape ]
        nb += int( np.prod( count ) )
        if all( c == 1 for c in count ):
            return nb
        l += 1


class _GridMajorant( Aggregate ):
    """the majorant of one node: `w <= wa . y + wb`"""
    nb_dims : CtShapeVar
    dim     : Axis[ "nb_dims" ]
    wa      : RealTensor[ "dim" ]
    wb      : RealTensor
