import jax.numpy as jnp

from loom.tensor import CtShapeVar
from loom.tensor import ShapeVar
from loom.tensor import Tensor
from loom.tensor import RealTensor
from loom.tensor import Axis
from loom.util import ComputedAttribute

from .Distribution import Distribution


class ProjectedSumOfTents( Distribution ):
    """1D TENTS that are the projections of shared points on a direction per item of a batch -- e.g. one per angle of a
    tomography: `points` `[ n, proj_dim ]`, `half_widths` `[ n ]`, `weights` `[ n ]` the masses, `normal` `[ proj_dim ]` per
    item. Tent i on the line has the density weights_i ( 1 - |y - s_i| / w_i )_+ / w_i, s_i = points_i . normal,
    w_i = half_widths_i. A continuous distribution: its transport against an `Image` ( `ContinuousOtPlan1d` ) is differentiable,
    with no kink when two tents cross -- what diracs lack.

    = Internals

    The twin of `ProjectedSumOfDiracs`, with a width: the projected positions are computed on the fly ( never an
    `[ nb_angles, n ]` array ), the points, half-widths and masses are SHARED across the batch, and their gradients are
    accumulated over the items by atomics ( `ProjectedSumOfTents.h` ). Its C++ struct emits the 3 n breakpoints of a
    view ( `fill_breakpoints` ), from which `ot1d::TentStream` makes linear pieces for the generic 1D transport
    ( `ot1d/transport_1d.h` ).
    """

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar     # the OT ( projected ) space: 1D
    proj_dims        : CtShapeVar     # the ambient space the points live in

    num_dirac        : Axis[ "nb_diracs" ]
    proj_dim         : Axis[ "proj_dims" ]

    points           : RealTensor[ "num_dirac", "proj_dim" ]
    half_widths      : RealTensor[ "num_dirac" ]
    normal           : RealTensor[ "proj_dim" ]
    weights          : RealTensor[ "num_dirac" ]

    current_mass     : ComputedAttribute[ RealTensor, ( "weights" ) ]

    def __init__( self, points, half_widths, normal, weights = None, target_mass = 1.0, proj_dims = 2, **kwargs ):
        self.__base_init__( points = points, half_widths = half_widths, normal = normal, weights = weights,
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
            return ProjectedSumOfTents(
                nb_diracs = self.nb_diracs.value,
                proj_dims = self.proj_dims.value,
                points = self.points,
                half_widths = self.half_widths,
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
        return ProjectedSumOfTents(
            points = self.points,
            half_widths = self.half_widths,
            normal = Tensor.wrap( self.normal.value[ index ], [ self.proj_dim.name ] ),
            weights = self.weights if self.weights.is_defined else None,
        )
