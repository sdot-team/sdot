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

    # always unidimensional ?
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
        # forward_args
        forward_args = [ "MF distance", "MF barycenters", "MF potentials" ]
        if f.always_1d or g.always_1d:
            forward_args += [ "MF cuts" ]
        for d_name, d_data in [ ( "f", f ), ( "g", g ) ]:
            for t_name, _ in d_data.tensor_list( True ):
                forward_args.append( f"NF { d_name }_{ t_name }" )

        # backward_args
        backward_args = [ "NF barycenters", "NF potentials" ]
        if f.always_1d or g.always_1d:
            backward_args += [ "NF cuts" ]
        backward_args += [ "NF grad_distances", "NF grad_barycenters", "NF grad_potentials", "NF grad_cuts" ]
        for d_name, d_data in [ ( "f", f ), ( "g", g ) ]:
            for t_name, _ in d_data.tensor_list( True ):
                backward_args.append( f"NF { d_name }_{ t_name }" )
        for d_name, d_data in [ ( "f", f ), ( "g", g ) ]:
            for t_name, _ in d_data.tensor_list( True ):
                backward_args.append( f"NF grad_{ d_name }_{ t_name }" )

        #
        input_instances = ""
        for d_name, d_data in [ ( "f", f ), ( "g", g ) ]:
            args = []
            for t_name, t_data in d_data.tensor_list( True ):
                args.append( f".{ t_name } = tensor_view_{ t_data.ndim }( { d_name }_{ t_name } )" )
            input_instances += f"{ d_data.__class__.__name__ }<const TF,Arch> { d_name }{{ { str.join( ", ", args ) } }}; "

        #
        grad_output_instances = ""
        for d_name, d_data in [ ( "f", f ), ( "g", g ) ]:
            args = []
            for t_name, t_data in d_data.tensor_list( True ):
                args.append( f".{ t_name } = tensor_view_{ t_data.ndim }( grad_{ d_name }_{ t_name } )" )
            grad_output_instances += f"{ d_data.__class__.__name__ }<TF,Arch> grad_{ d_name }{{ { str.join( ", ", args ) } }}; "

        includes = [ f"#include <sdot/{ g.__class__.__name__ }.h>" ]

        m = {
            "GRAD_OUTPUT_INSTANCES": grad_output_instances,
            "INPUT_INSTANCES": input_instances,
            "FORWARD_ARGS": str.join( ", ", forward_args ),
            "BACKWARD_ARGS": str.join( ", ", backward_args ),
            "SDOT_INCLUDES": str.join( "\n", includes ),
            "SDOT_CT_DIM": ct_dim,
        }

        return driver.cpp_src( m, """
            #include <sdot/nanobind_wrappers.h>
            #include <sdot/cpu/w2_distance.h>
            SDOT_INCLUDES

            namespace nb = nanobind;
            using namespace sdot;

            using NA = nanobind::device::SDOT_NANOBIND_ARCH;
            using TF = SDOT_SCALAR_TYPE;

            using Arch = ArchFor<NA>::type;
            using NF = nb::ndarray<const TF,NA>;
            using MF = nb::ndarray<TF,NA>;

            // Forward function that works with both 1D and 2D arrays
            void forward( FORWARD_ARGS ) {
                INPUT_INSTANCES
                w2_distance( f, g, tensor_view_1( distance ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ) );
            }

            void backward( BACKWARD_ARGS ) {
                GRAD_OUTPUT_INSTANCES
                INPUT_INSTANCES
                w2_distance_backward(
                              tensor_view_1( grad_distances ),
                              tensor_view_2( grad_barycenters ),
                              tensor_view_2( barycenters ),
                              tensor_view_2( potentials ),
                              tensor_view_3( cuts ), f, g, grad_f, grad_g );
            }

            #define MK_MOD( NAME ) NB_MODULE( NAME, m )

            MK_MOD( SDOT_BINDING_NAME ) {
                m.def( "backward", &backward );
                m.def( "forward", &forward );
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
