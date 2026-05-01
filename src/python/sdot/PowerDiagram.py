from .aggregate import aggregate, Workspace, Tensor, Return
from .distributions.Distribution import Distribution
from .driver import driver

from .Cell import BatchOfCell
from .Plotter import Plotter
from .Norm2 import Norm2
from .Bsp import Bsp

from typing import TYPE_CHECKING


@aggregate
class PowerDiagram:
    """

    """

    positions : Tensor( "nb_vertices", "dim", ct_axes = "dim" ) # vertex index ->
    weights : Tensor( "nb_vertices" ) # vertex index ->
    norm : Norm2
    bsp : Bsp

    if TYPE_CHECKING:
        max_of_nb_vertices : int
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
            points = Return( Tensor( "nb_points[]", "dim" ), max_of_nb_points = reservation ),
            faces = Return( Tensor( "nb_faces[]" ), max_of_nb_faces = reservation ),
            cells = Workspace( BatchOfCell,
                max_of_nb_vertices = 50,
                max_of_nb_edges = 50,
                max_of_nb_cuts = 50,
                batch_size = 1,
                dim = self.dim
            ),
            power_diagram = self
        )

    def newton_dir( self, g: Distribution ):
        """ On a la pos """
        res = driver.call( "newton_dir", "sdot/PowerDiagram/newton_dir.h", res = Return( Tensor, [ ] ), pd = self )
        info( res )
