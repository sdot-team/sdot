from .distributions.helpers.distribution_methods import _collect_attributes
from .distributions.helpers.ListOfTensorFields import ListOfTensorFields
from .distributions.helpers.TensorField import TensorField

from .distributions.BatchOfDistributions import BatchOfDistributions
from .distributions.SumOfWeightedDiracs import SumOfWeightedDiracs
from .distributions.Distribution import Distribution

from .BatchOf1dOtPlans import BatchOf1dOtPlans
from .BatchOfOtPlans import BatchOfOtPlans
from .OtPlan1d import OtPlan1d
from .OtPlan import OtPlan
from .Bsp import Bsp

from .driver import driver, encode_base62, tensor_conv_for

from typing import TYPE_CHECKING


def ot_plan( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ) -> OtPlan | OtPlan1d | BatchOfOtPlans | BatchOf1dOtPlans:
    # ensure batch version
    if isinstance( f, Distribution ) and isinstance( g, Distribution ):
        res = ot_plan( f.batch_version( 1 ), g.batch_version( 1 ) )
        assert isinstance( res, ( BatchOfOtPlans, BatchOf1dOtPlans ) )
        return res.unbatch()
    if isinstance( f, Distribution ):
        return ot_plan( f.batch_version( g.batch_size ), g )
    if isinstance( g, Distribution ):
        return ot_plan( f, g.batch_version( f.batch_size ) )

    # always unidimensional ?
    if f.is_an_unidimensional_verion and g.is_an_unidimensional_verion:
        return ot_plan( f.multidimensional_version(), g.multidimensional_version() ).unidimensional_version()
    if f.is_an_unidimensional_verion:
        return ot_plan( f.multidimensional_version(), g )
    if g.is_an_unidimensional_verion:
        return ot_plan( f, g.multidimensional_version() )

    # ensure `f` is a BatchOfSumOfWeightedDiracs, even if it means swapping `f` and `g`
    if not isinstance( f, SumOfWeightedDiracs.BatchVersion ):
        if isinstance( g, SumOfWeightedDiracs.BatchVersion ):
            return ot_plan( g, f )
        raise RuntimeError( "TODO: handle cases where f and g are both _not_ SumOfWeightedDiracs" )

    # check dims are the same
    if f.dim != g.dim:
        raise RuntimeError( f"f and g are not of same dim ({ f.dim } and { g.dim })" )

    # special case: 1d
    if f.dim == 1:
        return _ot_plan_1d( f, g )

    # generic case
    return _ot_plan_nd( f, g )


def _ot_plan_nd( batch_of_f : BatchOfDistributions, batch_of_g : BatchOfDistributions ) -> BatchOfOtPlans:
    """
    """
    if TYPE_CHECKING:
        from . import BatchOfSumOfWeightedDiracs
        assert isinstance( batch_of_f, BatchOfDistributions )
        assert isinstance( batch_of_f, BatchOfSumOfWeightedDiracs )

    #
    bindings = None

    # for now, we make an ot_plan for each batch
    for batch_index in range( batch_of_f.batch_size ):
        f = batch_of_f.batch_item( batch_index )
        g = batch_of_g.batch_item( batch_index )
        assert isinstance( g, Distribution )

        if bindings is None:
            bindings = _nd_bindings( f, g )

        bsp = Bsp( f.positions, f.weights )
        solve_nd( bindings, bsp, g )

        # return driver.plan_nd( bindings, f, g )

    distances = []
    barycenters = []
    potentials = []
    cuts = []
    return BatchOfOtPlans( distances, barycenters, potentials, cuts )

def solve_nd( bindings, bsp : Bsp, g : Distribution ):
    from .distributions.helpers.distribution_methods import flat_tensor_list, unflat_tensor_list
    flatten_g_content = flat_tensor_list( g )

    assert len( bsp.items ) == 1

    binding_inputs = []
    unflat_tensor_list( g, binding_inputs, list( flatten_g_content ) )


    # arg0: std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>,
    # arg1: sdot.bindings.bsp_FP64_2_cpu.Bsp,
    # arg2: ndarray[dtype=float64, device='cpu', writable=False],
    # arg3: ndarray[dtype=float64, device='cpu', writable=False],
    # arg4: collections.abc.Sequence[ndarray[dtype=float64, device='cpu', writable=False]]

    # Invoked with types: str, sdot.bindings.bsp_FP64_2_cpu.Bsp, jaxlib._jax.ArrayImpl, jaxlib._jax.ArrayImpl, list

    bindings.write_vtk( "pouet.vtk", bsp.items[ 0 ], *binding_inputs )
    # raise NotImplementedError


def _ot_plan_1d( f : BatchOfDistributions, g : BatchOfDistributions ):
    p_func, g_func, includes = type( g ).BaseVersion.primitive_function( g )

    dylib_name = f"ot_plan_{ encode_base62( "||".join( [ p_func, g_func ] + includes ) ) }_1d_{ driver.normalized_dtype }_{ driver.normalized_device_type }"

    def src_func():
        # inputs of backward_args and forward_args
        backward_args = []
        forward_args = []
        for d_name, d_data in [ ( "f", f ), ( "g", g ) ]:
            for a_name, a_data in _collect_attributes( type( d_data ) ):
                if isinstance( a_data, ListOfTensorFields ):
                    backward_args.append( ( "const std::vector<AF> &", f"{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )
                    forward_args.append( ( "const std::vector<AF> &", f"{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )
                if isinstance( a_data, TensorField ):
                    backward_args.append( ( "AF", f"{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )
                    forward_args.append( ( "AF", f"{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )

        # outputs of backward_args and forward_args
        backward_args += [ ( "AF", "distances", 1 ), ( "AF", "barycenters", 3 ), ( "AF", "potentials", 2 ), ( "AF", "cuts", 3 ),
                           ( "AF", "grad_distances", 1 ), ( "AF", "grad_barycenters", 3 ), ( "AF", "grad_potentials", 2 ), ( "AF", "grad_cuts", 3 ) ]
        forward_args += [ ( "MF", "distances", 1 ), ( "MF", "barycenters", 3 ), ( "MF", "potentials", 2 ), ( "MF", "cuts", 3 ) ]

        for d_name, d_data in [ ( "f", f ), ( "g", g ) ]:
            for a_name, a_data in _collect_attributes( type( d_data ) ):
                if isinstance( a_data, ListOfTensorFields ):
                    backward_args.append( ( "std::vector<MF> &", f"grad_{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )
                if isinstance( a_data, TensorField ):
                    backward_args.append( ( "MF", f"grad_{ d_name }_{ a_name }", a_data._rank( d_data ) or -1 ) )

        # tensor_conv
        backward_tensor_conv = str.join( " ", [ tensor_conv_for( t, n, r ) for t, n, r in backward_args ] )
        forward_tensor_conv = str.join( " ", [ tensor_conv_for( t, n, r ) for t, n, r in forward_args ] )

        m = {
            "BACKWARD_TENSOR_CONV": backward_tensor_conv,
            "FORWARD_TENSOR_CONV": forward_tensor_conv,
            "BACKWARD_ARGS": str.join( ", ", [ t + " _" + n for t, n, _ in backward_args ] ),
            "FORWARD_ARGS": str.join( ", ", [ t + " _" + n for t, n, _ in forward_args ] ),
            "PRIMITIVE_FUNC": p_func,
            "PRIMITIVE_GRAD": g_func,
            "SDOT_INCLUDES": str.join( "\n", [ f"#include <{ include }>" for include in includes ] ),
        }

        return driver.cpp_src( m, """
            #include <sdot/nanobind_wrappers.h>
            #include <nanobind/stl/vector.h>
            #include <sdot/ot_plan_1d.h>
            SDOT_INCLUDES

            namespace nb = nanobind;
            using namespace sdot;

            using NA = nanobind::device::SDOT_NANOBIND_ARCH;
            using TF = SDOT_SCALAR_TYPE;

            using Arch = ArchFor<NA>::type;
            using AF = nb::ndarray<const TF,NA>; // const array
            using MF = nb::ndarray<TF,NA>; // mutable array

            // Forward function that works with both 1D and 2D arrays
            void forward( FORWARD_ARGS ) {
                FORWARD_TENSOR_CONV
                ot_plan_1d_forward( f_positions, f_weights, PRIMITIVE_FUNC, distances, barycenters, potentials, cuts );
            }

            void backward( BACKWARD_ARGS ) {
                BACKWARD_TENSOR_CONV
                ot_plan_1d_backward( f_positions, f_weights, PRIMITIVE_FUNC, distances, barycenters, potentials, cuts, grad_distances, grad_barycenters, grad_potentials, grad_cuts, grad_f_positions, grad_f_weights, PRIMITIVE_GRAD );
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


def make_apply_fn( f: BatchOfDistributions, g: BatchOfDistributions ):
    """
    Returns a callable (*flat_inputs) -> (distances, barycenters, potentials, cuts)
    suitable for torch.autograd.gradcheck or jax.test_util.check_grads.

    flat_inputs is the concatenation of f.flat_tensor_list() and g.flat_tensor_list().
    The function rebuilds f and g from the flat inputs by walking _collect_attributes,
    then calls ot_plan internally.
    """
    def _rebuild( dist, fields, inp ):
        new_d = object.__new__( type( dist ) )
        new_d.__dict__.update( { k: v for k, v in dist.__dict__.items() if not k.startswith( '_' ) } )
        for name, field in fields:
            if isinstance( field, ListOfTensorFields ):
                count = getattr( dist, field.main_axis_name )
                orig = getattr( dist, name )
                tensors = [ inp.pop( 0 ) for _ in range( count ) ]
                if orig is not None:
                    setattr( new_d, name, tensors )
            else:
                t = inp.pop( 0 )
                if getattr( dist, name ) is not None:
                    setattr( new_d, name, t )
        return new_d

    f_fields = [ ( name, field ) for name, field in _collect_attributes( type( f ) )
                 if isinstance( field, ( TensorField, ListOfTensorFields ) ) ]
    g_fields = [ ( name, field ) for name, field in _collect_attributes( type( g ) )
                 if isinstance( field, ( TensorField, ListOfTensorFields ) ) ]

    def apply_fn( *flat_inputs ):
        inp = list( flat_inputs )
        new_f = _rebuild( f, f_fields, inp )
        new_g = _rebuild( g, g_fields, inp )
        plan = ot_plan( new_f, new_g )
        return plan.distances # , plan.barycenters, plan.potentials, plan.cuts

    return apply_fn


def distances( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    """ Squared Wasserstein distances """
    return ot_plan( f, g ).distances

def distance( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    """ Squared Wasserstein distance """
    d = distances( f, g )
    if d.ndim == 1:
        if d.shape[ 0 ] != 1:
            raise RuntimeError( "sdot.distance works only for batch_size = 1" )
        return d[ 0 ]
    assert d.ndim == 0
    return d.item()


def barycenters( f: Distribution | BatchOfDistributions, g: Distribution | BatchOfDistributions ):
    return ot_plan( f, g ).barycenters

