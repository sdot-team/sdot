from sdot.distributions.Distribution import Distribution

import loom
from loom.tensor import CtShapeVar
from loom.tensor import ShapeVar
from loom.tensor import RealTensor
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

    `sdot::continuous_ot_1d` ( `ContinuousOtPlan1d.h` ). Per view: the 3 n breakpoints sorted, the slope, density and
    cumulated mass at each by scans, then the sweep in CHUNKS -- each chunk finds where its pieces start in the image by
    a binary search and runs the generic 1D transport ( `ot1d/transport_1d.h` ) from there, a scan of the chunk totals
    making its sums global -- and the moments of each tent from the sums at its breakpoints. The transport is exact: a
    linear source piece against a constant target piece has a quadratic map, integrated by a 3-point Gauss-Legendre
    rule. On the host, one view per thread; on a card, the views by chunks, each by device-wide launches ( a CUB
    segmented sort, ONE scan for the three running quantities ). The adjoint runs the same transport and scatters
    d cost / d s_i, d cost / d w_i onto the shared points and half-widths by atomics.

    `with_gradient = True`: the gradient of `cost.sum()` with respect to the points and the half-widths, in the SAME
    pass ( `points_grad`, `half_widths_grad` ) -- half the work of `value_and_grad`, whose adjoint runs the transport
    again; for a `jax.custom_vjp` ( the call itself is then not differentiated ).

    `method = "grid"`: the tents' density taken as uniform inside each cell of a fine grid, `scale` cells per image bin
    ( `ContinuousOtPlan1dGrid.h`: an exact deposit of each tent into the cells, the sort-free transport of
    `sdot.projected1d` per view, the gradients by parts ). No sort: O( n x cells per tent ) per view. An error that
    vanishes as `scale` grows; the barycenters, not the potentials. The image's bins must be uniform.
    """

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar

    num_dirac        : Axis[ "nb_diracs" ]

    src_dist         : Distribution
    dst_dist         : Distribution

    barycenters      : RealTensor[ "num_dirac" ]
    potentials       : RealTensor[ "num_dirac" ]
    cost             : RealTensor

    def __init__( self, src_dist, dst_dist, with_barycenters = False, with_potentials = False, with_gradient = False,
                  method = "exact", scale = 8 ):
        if method not in ( "exact", "grid" ):
            raise ValueError( f"unknown method { method !r} ( exact, grid )" )
        if method == "grid" and with_potentials:
            raise NotImplementedError( "the grid method gives no potentials ( the cost, the barycenters, the gradients )" )
        self._method = method
        self._scale = int( scale )
        self._with_barycenters = bool( with_barycenters )
        self._with_potentials = bool( with_potentials )
        self._with_gradient = bool( with_gradient )
        if with_gradient and not hasattr( src_dist, "half_widths" ):
            raise NotImplementedError( "`with_gradient` is for tents with a half-width each ( `ProjectedSumOfTents` ): differentiate the cost with jax" )
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

    def update_outputs( self ):
        """`sdot::continuous_ot_1d` ( `ContinuousOtPlan1d.h` ): on the host, the views split between the threads; on a
        card, the views by chunks, each by device-wide launches ( segmented sort, scans, sweep by chunks ). Its memory
        is taken from the call's pool, at the size the call knows ( `args.allocator` )."""
        written = [ "cost", "nb_diracs" ] + ( [ "barycenters" ] if self._with_barycenters else [] ) \
                                          + ( [ "potentials" ] if self._with_potentials else [] )
        kp = loom.kernel_parameters_for( "update_outputs_ContinuousOtPlan1d" )
        # the exact transport, or the grid method ( `ContinuousOtPlan1dGrid.h` ) and its cells per bin
        fn, extra = ( "continuous_ot_1d", "" ) if self._method == "exact" else ( "continuous_ot_1d_grid", f", { self._scale }" )
        if self._with_gradient:
            # the forward and the gradient of cost.sum() in the same pass ( no adjoint: see `with_gradient` )
            src = self.src_dist
            bits = dict( size = 8 * src.points.value.dtype.itemsize )     # in the points' float
            points_grad = RealTensor[ src.num_dirac, src.proj_dim, bits ]()
            half_widths_grad = RealTensor[ src.num_dirac, bits ]()
            kp.ffi_call(
                FfiCode.inline( f"sdot::{ fn }( queue, batch_axes, args.outputs.plan, nullptr, args.allocator{ extra }, "
                                "sdot::ot1d_coop::SumGrad{ args.outputs.points_grad, args.outputs.half_widths_grad } );", allocator = True ),
                plan = loom.out( self, writes = written ),
                points_grad = loom.out( points_grad ),
                half_widths_grad = loom.out( half_widths_grad ),
                has_dynamic_capacity = False,
            )
            self.points_grad, self.half_widths_grad = points_grad, half_widths_grad
            return
        kp.ffi_call(
            FfiCode.inline( f"sdot::{ fn }( queue, batch_axes, args.outputs.plan, nullptr, args.allocator{ extra } );", allocator = True ),
            FfiCode.inline( "args.outputs.plan.src_dist.zero_grads( queue, args.grad_of_outputs.plan.src_dist );\n"
                            "    DECAYED_TYPE_OF( args.outputs.plan )::update_outputs_computed_grads( args.computed_grads.from_plan_cost.plan );\n"
                            f"    sdot::{ fn }( queue, batch_axes, args.outputs.plan, args.grad_of_outputs.plan, args.allocator{ extra } );",
                            allocator = True ),
            plan = loom.out( self, writes = written ),
            has_dynamic_capacity = False,
        )
