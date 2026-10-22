import jax.numpy as jnp

from loom.tensor import CtShapeVar
from loom.tensor import ShapeVar
from loom.tensor import Tensor
from loom.tensor import RealTensor
from loom.tensor import Axis
from loom.util import ComputedAttribute

from .Distribution import Distribution


class ProjectedSumOfAnisotropicTents( Distribution ):
    """1D TENTS of different widths at different directions: the projections of ELLIPTIC objects. Like `ProjectedSumOfTents`, with a
    matrix `matrices[ i ]` `[ proj_dim, proj_dim ]` per tent instead of a half-width: tent i on the line of direction `normal` is
    centered at s_i = points_i . normal, with the half-width w_i = | matrices_i^T normal |, so that an isotropic tent is
    `matrices_i = w_i * identity`. It is what an object whose shape is the image of a disk by `matrices_i` projects to
    ( a transformed disk is an ellipse ). Differentiable with respect to `points` and `matrices`, through
    `ContinuousOtPlan1d` ( the exact method, on the host ). `ProjectedSumOfTents` is faster when all the tents are round.

    = Internals

    Its C++ struct ( `ProjectedSumOfAnisotropicTents.h` ) answers the same contract as `ProjectedSumOfTents`, with a half-width
    that depends on the item's direction ( computed from the matrix on the fly ) and its gradient `n ( matrices^T n )^T / w`
    scattered onto the matrix by atomics. The card's kernels assume a half-width SHARED by the items: they refuse this type.
    """

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar     # the OT ( projected ) space: 1D
    proj_dims        : CtShapeVar     # the ambient space the points live in

    num_dirac        : Axis[ "nb_diracs" ]
    proj_dim         : Axis[ "proj_dims" ]
    proj_dim_b       : Axis[ "proj_dims" ]

    points           : RealTensor[ "num_dirac", "proj_dim" ]
    matrices         : RealTensor[ "num_dirac", "proj_dim", "proj_dim_b" ]
    normal           : RealTensor[ "proj_dim" ]
    weights          : RealTensor[ "num_dirac" ]

    current_mass     : ComputedAttribute[ RealTensor, ( "weights" ) ]

    def __init__( self, points, matrices, normal, weights = None, target_mass = 1.0, proj_dims = 2, **kwargs ):
        self.__base_init__( points = points, matrices = matrices, normal = normal, weights = weights,
                            target_mass = target_mass, nb_dims = 1, proj_dims = proj_dims, **kwargs )

    def normalized_version( self, nb_dims = None ):
        # the weights only ( SHARED across the batch, as for `ProjectedSumOfDiracs` ), in the points' float: a float64
        # weight or mass would promote float32 points ( and their kernels ) to float64
        mass = self.mass
        if self.target_mass.is_defined:
            dt = self.points.value.dtype
            bits = dict( size = 8 * dt.itemsize )
            if self.weights.is_defined:
                weights = self.target_mass / mass * self.weights
            else:
                weights = RealTensor[ self.num_dirac, bits ].full( self.target_mass / self.nb_diracs.value )
            return ProjectedSumOfAnisotropicTents(
                nb_diracs = self.nb_diracs.value,
                proj_dims = self.proj_dims.value,
                points = self.points,
                matrices = self.matrices,
                normal = self.normal,
                weights = weights,
                current_mass = RealTensor[ bits ]( jnp.asarray( self.target_mass.value, dt ) ),
                batch_axes = self.batch_axes,
            )
        return self

    def _update_current_mass( self ):
        if self.weights.is_defined:
            self.current_mass = self.weights.sum()
        else:
            self.current_mass = self.nb_diracs.value

    def batch_slice( self, index ):
        # only `normal` varies with the item
        if not self.batch_axes or not self.normal.is_defined:
            return None
        return ProjectedSumOfAnisotropicTents(
            points = self.points,
            matrices = self.matrices,
            normal = Tensor.wrap( self.normal.value[ index ], [ self.proj_dim.name ] ),
            weights = self.weights if self.weights.is_defined else None,
        )
