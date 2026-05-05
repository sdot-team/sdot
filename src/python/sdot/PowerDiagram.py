from .aggregate import aggregate, Workspace, Tensor, Return
from .distributions.Distribution import Distribution
from .driver import driver

from .CellWorkspace import BatchOfCellWorkspace
from .MatrixTerms import MatrixTerms
from .Cell import BatchOfCell
from .Plotter import Plotter
from .Norm2 import Norm2
from .Bsp import Bsp

from typing import TYPE_CHECKING


@aggregate
class PowerDiagram:
    """

    """

    positions : Tensor( "nb_vertices", "dim", ct_axes = [ "dim" ] ) # vertex index ->
    weights : Tensor( "nb_vertices" ) # vertex index ->
    ground_metric : Norm2
    bsp : Bsp

    if TYPE_CHECKING:
        nb_vertices : int
        dim : int

    def __init__( self, positions, weights = None, ground_metric = None ):
        self.positions = positions
        self.weights = weights

        if ground_metric is None:
            ground_metric = Norm2( driver.empty( [ 0, self.dim + 1, self.dim + 1 ] ) )
        self.ground_metric = ground_metric
        info( self.ground_metric )

        self.bsp = Bsp.make_from( self.positions, self.weights )

    def plot( self, target_distribution, plotter = None ):
        with Plotter( plotter ) as p:
            p.plot_mesh( *self.vtk_cell_faces( target_distribution ) )

    def vtk_cell_faces( self, target_distribution ):
        reservation = 64 * self.nb_vertices # TODO better approx

        return driver.call( "vtk_cell_faces", "sdot/PowerDiagram/vtk_cell_faces.h",
            points = Return( Tensor( "nb_points[]", "dim" ), max_of_nb_points = reservation, dim = self.dim ),
            faces = Return( Tensor( "nb_faces[]" ), max_of_nb_faces = reservation ),
            cell_workspace = Workspace( BatchOfCellWorkspace,
                max_of_nb_indices_to_remove = 256,
                max_of_nb_map_items = 256,
                max_of_reservation = 256,
                max_of_nb_links = 512,
                batch_size = 1,
            ),
            cells = Workspace( BatchOfCell,
                max_of_nb_vertices = 50,
                max_of_nb_edges = 50,
                max_of_nb_cuts = 50,
                batch_size = 1,
                dim = self.dim
            ),
            target_distribution = target_distribution,
            power_diagram = self,

            grad = False
        )

    def adjust_weights( self, dirac_masses, target_distribution ):
        new_weights, barycenter, distance = driver.call( "adjust_weights", "sdot/PowerDiagram/adjust_weights.h",
            new_weights = Return( Tensor( "nb_vertices" ), nb_vertices = self.nb_vertices ),
            barycenter = Return( Tensor( "nb_vertices", "dim" ), nb_vertices = self.nb_vertices, dim = self.dim ),
            distance = Return( Tensor() ),
            cell_workspace = Workspace( BatchOfCellWorkspace,
                max_of_nb_indices_to_remove = 256,
                max_of_nb_map_items = 256,
                max_of_reservation = 256,
                max_of_nb_links = 512,
                batch_size = 1,
                # dim = self.dim
            ),
            cells = Workspace( BatchOfCell,
                max_of_nb_vertices = 50,
                max_of_nb_edges = 50,
                max_of_nb_cuts = 50,
                batch_size = 1,
                dim = self.dim
            ),
            matrix_terms = Workspace( MatrixTerms,
                max_of_nb_matrix_terms = 10 * self.nb_vertices,
                nb_vector_terms = self.nb_vertices
            ),
            target_distribution = target_distribution,
            dirac_masses = dirac_masses,
            power_diagram = self,

            max_iteration_count = 2000,
            verbosity = 2,

            grad = False
        )

        self.weights = new_weights

        return distance, barycenter
