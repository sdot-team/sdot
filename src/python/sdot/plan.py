from .distributions.BatchOfDistributions import BatchOfDistributions
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution
from .BatchOf1dOtPlans import BatchOf1dOtPlans
from .BatchOfOtPlans import BatchOfOtPlans
from .OtPlan1d import OtPlan1d
from .OtPlan import OtPlan
from .driver import driver


def plan( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions, _check_1d = True ) -> OtPlan | OtPlan1d | BatchOfOtPlans | BatchOf1dOtPlans:
    # ensure batch
    if isinstance( f, Distribution ) and isinstance( g, Distribution ):
        return plan( f.batch_version( 1 ), g.batch_version( 1 ) ).unbatch()
    if isinstance( f, Distribution ):
        return plan( f.batch_version( g.batch_size ), g )
    if isinstance( g, Distribution ):
        return plan( f, g.batch_version( f.batch_size ) )

    # always unidimensional
    if _check_1d and f.always_1d:
        return plan( f, g, _check_1d = False ).unidimensionnal_version()

    # ensure `f` is a BatchOfSumOfWeightedDiracs, even if it means swapping `f` and `g`
    if not isinstance( f, SumOfWeightedDiracs.batch_class() ):
        if isinstance( g, SumOfWeightedDiracs.batch_class() ):
            return plan( g, f )
        raise RuntimeError( "TODO: handle cases where f and g are both _not_ SumOfWeightedDiracs" )

    #
    ct_dim = f.dim if f.dim <= 4 else -1
    f_name = f.__class__.__name__
    g_name = g.__class__.__name__

    dylib_name = f"ot_plan_{ f_name }_{ g_name }_{ ct_dim }d_{ driver.normalized_dtype }_{ driver.normalized_device_type }"

    def src_func():
        return driver.cpp_src( { "SDOT_CT_DIM": ct_dim }, """
            #include <sdot/nanobind_wrappers.h>
            #include <sdot/cpu/w2_distance.h>

            namespace nb = nanobind;
            using namespace sdot;

            using NA = nanobind::device::SDOT_NANOBIND_ARCH;
            using TF = SDOT_SCALAR_TYPE;

            using Arch = ArchFor<NA>::type;
            using NF = nb::ndarray<const TF,NA>;
            using MF = nb::ndarray<TF,NA>;

            // Forward function that works with both 1D and 2D arrays
            void forward( NF dirac_xs, NF dirac_ws, NF point_xs, NF point_ys, MF distance, MF barycenters, MF potentials, MF cuts ) {
                BatchOfAffine1d<const TF,Arch> functions{ .xs = tensor_view_2( point_xs ), .ys = tensor_view_2( point_ys ) };
                BatchOfDiracSet<const TF,Arch> diracs{ .xs = tensor_view_3( dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( dirac_ws ) };
                w2_distance( diracs, functions, tensor_view_1( distance ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ) );
            }

            void backward(
                NF barycenters, NF potentials, NF cuts,
                NF dirac_xs, NF dirac_ws, NF point_xs, NF point_ys,
                NF grad_distances, NF grad_barycenters, NF grad_potentials, NF grad_cuts,
                MF grad_dirac_xs, MF grad_dirac_ws, MF grad_point_xs, MF grad_point_ys
            ) {
                BatchOfAffine1d<TF,Arch> grad_functions{ .xs = tensor_view_2( grad_point_xs ), .ys = tensor_view_2( grad_point_ys ) };
                BatchOfDiracSet<TF,Arch> grad_diracs{ .xs = tensor_view_3( grad_dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( grad_dirac_ws ) };
                BatchOfAffine1d<const TF,Arch> functions{ .xs = tensor_view_2( point_xs ), .ys = tensor_view_2( point_ys ) };
                BatchOfDiracSet<const TF,Arch> diracs{ .xs = tensor_view_3( dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( dirac_ws ) };
                w2_distance_backward( tensor_view_1( grad_distances ), tensor_view_2( grad_barycenters ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ), diracs, functions, grad_diracs, grad_functions );
            }

            #define MK_MOD( NAME ) NB_MODULE( NAME, m )

            MK_MOD( SDOT_BINDING_NAME ) {
                m.def( "forward", &forward,
                    nb::arg( "dirac_xs" ), nb::arg( "dirac_ws" ), nb::arg( "point_xs" ), nb::arg( "point_ys" ),
                    nb::arg( "distance" ), nb::arg( "barycenters" ), nb::arg( "potentials" ), nb::arg( "cuts" ),
                    "SDOT plan to get distance and barycenters with f = sum of diracs (1D), g = piecewise affine function"
                );

                m.def( "backward", &backward,
                    nb::arg( "barycenters" ), nb::arg( "potentials" ), nb::arg( "cuts" ),
                    nb::arg( "dirac_xs" ), nb::arg( "dirac_ws" ), nb::arg( "points_xs" ), nb::arg( "points_ys" ),
                    nb::arg( "grad_distance" ), nb::arg( "grad_barycenters" ), nb::arg( "grad_potentials" ), nb::arg( "grad_cuts" ),
                    nb::arg( "grad_dirac_xs" ), nb::arg( "grad_dirac_ws" ), nb::arg( "grad_points_xs" ), nb::arg("grad_points_ys"),
                    "SDOT W2 backward implementation"
                );
            }
        """ )

    # get the binding
    bindings = driver.import_bindings( dylib_name, src_func, [] )
    return driver.plan( bindings, f, g )


def distances( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    return plan( f, g ).distances

def distance( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    d = distances( f, g )
    if d.ndim == 1:
        if d.shape[ 0 ] != 1:
            raise RuntimeError( "sdot.distance works only for batch_size = 1" )
        return d[ 0 ]
    assert d.ndim == 0
    return d.item()


def barycenters( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    return plan( f, g ).barycenters
