from sdot.distributions.Distribution import Distribution

import loom
from loom.tensor import CtShapeVar
from loom.tensor import ShapeVar
from loom.tensor import RealTensor
from loom.tensor import IntTensor
from loom.tensor import Axis

from loom.compilation.FfiCode import FfiCode
from loom.util import Aggregate


class ContinuousOtPlan1d( Aggregate ):
    """THE TRANSPORT between two CONTINUOUS distributions on a line: today a `ProjectedSumOfTents` ( the source ) against
    an `Image` ( the target ), batched over the items of both ( e.g. the angles of a sinogram ), differentiable with
    respect to the tents' points and half-widths.

    `cost` is W2^2 ( both sides normalized to mass 1 ); `barycenters` ( b_i = int T t_i, `[ n ]` ) and `potentials` ( the
    derivative with respect to the mass of each tent, centered ) are stored only when asked for.

    = Internals

    Each item ( a view ) is one work-item: it fills the 3 n breakpoints of its tents, sorts them, and runs the generic 1D
    transport ( `ot1d/transport_1d.h` ) against the image's constant pieces, exactly ( a linear source piece against a
    constant target piece has a quadratic transport map, integrated by a 3-point Gauss-Legendre rule ). The adjoint
    runs the same sweep and scatters d cost / d s_i, d cost / d w_i onto the shared points and half-widths by atomics.
    The breakpoints and their sums live in per-thread scratch rows, never one per view.
    """

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar

    num_dirac        : Axis[ "nb_diracs" ]

    src_dist         : Distribution
    dst_dist         : Distribution

    barycenters      : RealTensor[ "num_dirac" ]
    potentials       : RealTensor[ "num_dirac" ]
    cost             : RealTensor

    def __init__( self, src_dist, dst_dist, with_barycenters = False, with_potentials = False ):
        self._with_barycenters = bool( with_barycenters )
        self._with_potentials = bool( with_potentials )
        src_dist = src_dist.normalized_version( nb_dims = 1 )
        dst_dist = dst_dist.normalized_version( nb_dims = 1 )
        self.__base_init__(
            nb_diracs = src_dist.nb_diracs.value,
            nb_dims = 1,
            src_dist = src_dist,
            dst_dist = dst_dist,
            batch_axes = src_dist.batch_axes,
        )
        self.update_outputs()

    def _scratch( self ):
        n = int( self.nb_diracs.value )
        kp = loom.kernel_parameters_for( "update_outputs_ContinuousOtPlan1d" )
        nt = kp.nb_threads( batch_axes = self.batch_axes, nb_local_bytes_per_thread = 3 * n * ( 8 + 8 + 4 + 4 * 8 ) )
        num_group = Axis( ShapeVar( nt ), name = "num_group" )
        num_bp = Axis( ShapeVar( 3 * n ), name = "num_breakpoint" )
        num_sum = Axis( ShapeVar( 4 ), name = "num_sum" )
        return {
            "bp_pos":    RealTensor[ num_group, num_bp ](),
            "bp_dslope": RealTensor[ num_group, num_bp ](),
            "bp_id":     IntTensor[ num_group, num_bp ](),
            "sums":      RealTensor[ num_group, num_bp, num_sum ](),
        }

    def update_outputs( self ):
        written = [ "cost", "nb_diracs" ] + ( [ "barycenters" ] if self._with_barycenters else [] ) \
                                          + ( [ "potentials" ] if self._with_potentials else [] )
        rows = "scratch.bp_pos( thread_index ), scratch.bp_dslope( thread_index ), scratch.bp_id( thread_index ), scratch.sums( thread_index )"
        geometry = dict( max_nb_threads = "return scratch.bp_pos.shape( 0 );" )
        kp = loom.kernel_parameters_for( "update_outputs_ContinuousOtPlan1d" )
        kp.ffi_call(
            FfiCode.per_item( code = f"outputs.plan( batch_index ).update_outputs( { rows } );", **geometry ),
            FfiCode.per_item( code = f"outputs.plan( batch_index ).update_outputs_bwd( grad_of_outputs.plan( batch_index ), { rows } );",
                              prologue = "outputs.plan.src_dist.zero_grads( queue, grad_of_outputs.plan.src_dist );",
                              computed_grads = "DECAYED_TYPE_OF( outputs.plan )::update_outputs_computed_grads( computed_grads.from_plan_cost.plan );",
                              **geometry ),
            plan = loom.out( self, writes = written ),
            **{ n: loom.scratch( t ) for n, t in self._scratch().items() },
            has_dynamic_capacity = False,
        )
