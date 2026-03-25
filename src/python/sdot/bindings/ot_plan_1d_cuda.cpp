#include "../../../cpp/cpu/w2_distance.h" // IWYU pragma: keep
#include "nanobind_wrappers.h"

#define CHECK_RANK( TENSOR, VALUE, HELP ) if ( ( TENSOR ).ndim() != VALUE ) throw std::invalid_argument( "rank of " #TENSOR " must be " #VALUE " (" #TENSOR HELP ")" )
#define CHECK_BATCH_SIZE( TENSOR ) if ( ( TENSOR ).shape( 0 ) != batch_size ) throw std::invalid_argument( "batch sizes of " #TENSOR " does not match" )
#define CHECK_DIM_1( TENSOR ) if ( ( TENSOR ).shape( 2 ) != 1 ) throw std::invalid_argument( "dim of " #TENSOR " must be 1" )

using namespace sdot;
namespace nb = nanobind;

using TF = SDOT_SCALAR_TYPE;

using NF = nb::ndarray<const TF,nb::device::cuda>;
using MF = nb::ndarray<TF,nb::device::cuda>;
// using TF = SDOT_CT_DIM;


// Forward function that works with both 1D and 2D arrays
static void forward( NF dirac_xs, NF dirac_ws, NF point_xs, NF point_ys, MF distance, MF barycenters, MF potentials, MF cuts ) {
    // ranks
    CHECK_RANK( barycenters, 3, "[ batch_index, dirac_index, dim ]" );
    CHECK_RANK( potentials, 2, "[ batch_index, dirac_index ]" );
    CHECK_RANK( distance, 1, "[ batch_index ]" );
    CHECK_RANK( cuts, 3, "[ batch_index, cut_index, bound_index ]" );

    CHECK_RANK( dirac_xs, 3, "[ batch_index, dirac_index, dim ]" );
    CHECK_RANK( dirac_ws, 2, "[ batch_index, dirac_index ]" );
    CHECK_RANK( point_xs, 2, "[ batch_index, point_index ]" );
    CHECK_RANK( point_ys, 2, "[ batch_index, point_index ]" );

    // batch_size
    const PI batch_size = dirac_xs.shape( 0 );
    CHECK_BATCH_SIZE( barycenters );
    CHECK_BATCH_SIZE( potentials );
    CHECK_BATCH_SIZE( distance );
    CHECK_BATCH_SIZE( dirac_ws );
    CHECK_BATCH_SIZE( point_xs );
    CHECK_BATCH_SIZE( point_ys );
    CHECK_BATCH_SIZE( cuts );

    // space dim
    CHECK_DIM_1( dirac_xs );

    ASSERT( 0 );
    // BatchOfAffine1d<const TF> functions{ .xs = tensor_view_2( point_xs ), .ys = tensor_view_2( point_ys ) };
    // BatchOfDiracSet<const TF> diracs{ .xs = tensor_view_3( dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( dirac_ws ) };
    // w2_distance( diracs, functions, tensor_view_1( distance ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ) );
}

static void backward(
    NF barycenters, NF potentials, NF cuts,
    NF dirac_xs, NF dirac_ws, NF point_xs, NF point_ys,
    NF grad_distances, NF grad_barycenters, NF grad_potentials, NF grad_cuts,
    MF grad_dirac_xs, MF grad_dirac_ws, MF grad_point_xs, MF grad_point_ys
 ) {
    // rank
    CHECK_RANK( grad_barycenters, 3, "[ batch_index, dirac_index, dim ]" );
    // CHECK_RANK( grad_potentials, 2, "[ batch_index, dirac_index ]" );
    CHECK_RANK( grad_distances, 1, "[ batch_index ]" );

    CHECK_RANK( barycenters, 3, "[ batch_index, dirac_index, dim ]" );
    CHECK_RANK( potentials, 2, "[ batch_index, dirac_index ]" );
    CHECK_RANK( cuts, 3, "[ batch_index, cut_index, bound_index ]" );

    CHECK_RANK( dirac_xs, 3, "[ batch_index, dirac_index, dim ]" );
    CHECK_RANK( dirac_ws, 2, "[ batch_index, dirac_index ]" );
    CHECK_RANK( point_xs, 2, "[ batch_index, point_index ]" );
    CHECK_RANK( point_ys, 2, "[ batch_index, point_index ]" );

    CHECK_RANK( grad_dirac_xs, 3, "[ batch_index, dirac_index, dim ]" );
    CHECK_RANK( grad_dirac_ws, 2, "[ batch_index, dirac_index ]" );
    CHECK_RANK( grad_point_xs, 2, "[ batch_index, point_index ]" );
    CHECK_RANK( grad_point_ys, 2, "[ batch_index, point_index ]" );

    // batch_size...
    const PI batch_size = dirac_xs.shape( 0 );
    CHECK_BATCH_SIZE( grad_barycenters );
    CHECK_BATCH_SIZE( grad_distances );
    CHECK_BATCH_SIZE( barycenters );
    CHECK_BATCH_SIZE( potentials );
    CHECK_BATCH_SIZE( dirac_ws );
    CHECK_BATCH_SIZE( point_xs );
    CHECK_BATCH_SIZE( point_ys );
    CHECK_BATCH_SIZE( cuts );

    CHECK_BATCH_SIZE( grad_dirac_xs );
    CHECK_BATCH_SIZE( grad_dirac_ws );
    CHECK_BATCH_SIZE( grad_point_xs );
    CHECK_BATCH_SIZE( grad_point_ys );

    ASSERT( 0 );
    // BatchOfAffine1d<TF> grad_functions{ .xs = tensor_view_2( grad_point_xs ), .ys = tensor_view_2( grad_point_ys ) };
    // BatchOfDiracSet<TF> grad_diracs{ .xs = tensor_view_3( grad_dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( grad_dirac_ws ) };
    // BatchOfAffine1d<const TF> functions{ .xs = tensor_view_2( point_xs ), .ys = tensor_view_2( point_ys ) };
    // BatchOfDiracSet<const TF> diracs{ .xs = tensor_view_3( dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( dirac_ws ) };
    // w2_distance_backward( tensor_view_1( grad_distances ), tensor_view_2( grad_barycenters ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ), diracs, functions, grad_diracs, grad_functions );
}

#define MK_MOD( NAME ) \
    NB_MODULE( NAME, m )

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

