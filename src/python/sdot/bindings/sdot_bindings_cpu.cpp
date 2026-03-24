#include "../../../cpp/cpu/w2_distance.h" // IWYU pragma: keep
#include "nanobind_wrappers.h"

#define CHECK_RANK( TENSOR, VALUE, HELP ) if ( ( TENSOR ).ndim() != VALUE ) throw std::invalid_argument( "rank of " #TENSOR " must be " #VALUE " (" #TENSOR HELP ")" )
#define CHECK_BATCH_SIZE( TENSOR ) if ( ( TENSOR ).shape( 0 ) != batch_size ) throw std::invalid_argument( "batch sizes of " #TENSOR " does not match" )
#define CHECK_DIM_1( TENSOR ) if ( ( TENSOR ).shape( 2 ) != 1 ) throw std::invalid_argument( "dim of " #TENSOR " must be 1" )

using namespace sdot;
namespace nb = nanobind;

using TF = SDOT_SCALAR_TYPE;
// using TF = SDOT_CT_DIM;


// Forward function that works with both 1D and 2D arrays
static void forward( nb::ndarray<const TF> dirac_xs, nb::ndarray<const TF> dirac_ws, nb::ndarray<const TF> point_xs, nb::ndarray<const TF> point_ys, nb::ndarray<TF> distance, nb::ndarray<TF> barycenters, nb::ndarray<TF> potentials, nb::ndarray<TF> cuts ) {
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

    Affine1d<const TF,1> functions{ .xs = tensor_view_2( point_xs ), .ys = tensor_view_2( point_ys ) };
    DiracSet<const TF,1> diracs{ .xs = tensor_view_3( dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( dirac_ws ) };
    w2_distance( diracs, functions, tensor_view_1( distance ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ) );
}

static void backward(
    nb::ndarray<const TF> barycenters, nb::ndarray<const TF> potentials, nb::ndarray<const TF> cuts,
    nb::ndarray<const TF> dirac_xs, nb::ndarray<const TF> dirac_ws, nb::ndarray<const TF> point_xs, nb::ndarray<const TF> point_ys,
    nb::ndarray<const TF> grad_distances, nb::ndarray<const TF> grad_barycenters, nb::ndarray<const TF> grad_potentials, nb::ndarray<const TF> grad_cuts,
    nb::ndarray<TF> grad_dirac_xs, nb::ndarray<TF> grad_dirac_ws, nb::ndarray<TF> grad_point_xs, nb::ndarray<TF> grad_point_ys
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

    Affine1d<TF,1> grad_functions{ .xs = tensor_view_2( grad_point_xs ), .ys = tensor_view_2( grad_point_ys ) };
    DiracSet<TF,1> grad_diracs{ .xs = tensor_view_3( grad_dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( grad_dirac_ws ) };
    Affine1d<const TF,1> functions{ .xs = tensor_view_2( point_xs ), .ys = tensor_view_2( point_ys ) };
    DiracSet<const TF,1> diracs{ .xs = tensor_view_3( dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( dirac_ws ) };
    w2_distance_backward( tensor_view_1( grad_distances ), tensor_view_2( grad_barycenters ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ), diracs, functions, grad_diracs, grad_functions );
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

