from sdot.object_with_tensors import object_with_tensors, TensorField
from .driver import driver, Workspace, Return, Tensor, CtKnown, Dyn
from .Cell import BatchOfCell
from .Plotter import Plotter
from .Norm2 import Norm2
from .Bsp import Bsp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .distributions.Distribution import Distribution


@object_with_tensors
class PowerDiagram:
    """

    """

    positions = TensorField( "nb_vertices", CtKnown( "dim" ) ) # vertex index ->
    weights = TensorField( "nb_vertices" ) # vertex index ->
    norm : Norm2 #
    bsp : Bsp


    if TYPE_CHECKING:
        nb_vertices_capacity : int
        nb_vertices : int
        dim : int

    def __init__( self, positions, weights = None, norm = Norm2 ):
        self.positions = positions
        self.weights = weights
        self.norm = norm

        self.bsp = Bsp.make_from( self.positions, self.weights )

    def plot( self, plotter = None ):
        with Plotter( plotter ) as p:
            p.plot_mesh( *self.vtk_cell_faces() )

    def vtk_cell_faces( self ):
        reservation = 1 # 10 * self.nb_vertices # TODO better approx

        return driver.call( "vtk_cell_faces", "sdot/PowerDiagram/vtk_cell_faces.h",
            points = Return( Tensor, [ Dyn( "nb_points", reservation ), self.dim ] ),
            faces = Return( Tensor, [ Dyn( "nb_faces", 1 ) ] ),
            cells = Workspace( BatchOfCell,
                nb_vertices_capacity = 50,
                nb_edges_capacity = 50,
                nb_cuts_capacity = 50,
                batch_size = 1,
                dim = self.dim
            ),
            power_diagram = self
        )

    def newton_dir( self, g: 'Distribution' ):
        """ On a la pos """
        res = driver.call( "newton_dir", "sdot/PowerDiagram/newton_dir.h", res = Return( Tensor, [ ] ), pd = self )
        info( res )
