#include "../../../cpp/cpu/w2_distance.h" // IWYU pragma: keep
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#define CHECK_RANK( TENSOR, VALUE, HELP ) if ( ( TENSOR ).ndim() != VALUE ) throw std::invalid_argument( "rank of " #TENSOR " must be " #VALUE " (" #TENSOR HELP ")" )
#define CHECK_BATCH_SIZE( TENSOR ) if ( ( TENSOR ).shape( 0 ) != batch_size ) throw std::invalid_argument( "batch sizes of " #TENSOR " does not match" )
#define CHECK_DIM_1( TENSOR ) if ( ( TENSOR ).shape( 2 ) != 1 ) throw std::invalid_argument( "dim of " #TENSOR " must be 1" )

namespace nb = nanobind;
using namespace sdot;
using TF = float;

using NA = nb::ndarray<TF>;

// Forward function that works with both 1D and 2D arrays
void ot_plan_to_piecewise_affine_1d( NA dirac_xs, NA dirac_ws, NA point_xs, NA point_ys, NA distance, NA barycenters ) {
    // ranks
    CHECK_RANK( dirac_xs, 3, "[ batch_index, dirac_index, dim ]" );
    CHECK_RANK( dirac_ws, 2, "[ batch_index, dirac_index ]" );
    CHECK_RANK( point_xs, 2, "[ batch_index, point_index ]" );
    CHECK_RANK( point_ys, 2, "[ batch_index, point_index ]" );

    // batch_size
    const PI batch_size = dirac_xs.shape( 0 );
    CHECK_BATCH_SIZE( dirac_ws );
    CHECK_BATCH_SIZE( point_xs );
    CHECK_BATCH_SIZE( point_ys );

    // space dim
    CHECK_DIM_1( dirac_xs );

    // helpers
    const auto cview2 = [&]( const NA &v ) { return TensorView<const TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };
    const auto view2 = [&]( NA &v ) { return TensorView<TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };
    const auto view1 = [&]( NA &v ) { return TensorView<TF,1>{ v.data(), batch_size }; };

    Affine1d<const TF,1> functions{ .xs = cview2( point_xs ), .ys = cview2( point_ys ) };
    DiracSet<const TF,1> diracs{ .xs = cview2( dirac_xs ), .ws = cview2( dirac_ws ) };
    w2_distance( diracs, functions, view1( distance ), view2( barycenters ) );
}

void backward_ot_plan_to_piecewise_affine_1d(
    NA grad_distances, NA grad_barycenters, NA dirac_xs, NA dirac_ws, NA point_xs, NA point_ys, NA barycenters,
    NA grad_dirac_xs, NA grad_dirac_ws, NA grad_point_xs, NA grad_point_ys ) {

    // rank
    CHECK_RANK( grad_barycenters, 3, "[ batch_index, dirac_index, dim ]" );
    CHECK_RANK( grad_distances, 1, "[ batch_index ]" );

    CHECK_RANK( barycenters, 3, "[ batch_index, dirac_index, dim ]" );
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
    CHECK_BATCH_SIZE( dirac_ws );
    CHECK_BATCH_SIZE( point_xs );
    CHECK_BATCH_SIZE( point_ys );

    CHECK_BATCH_SIZE( grad_dirac_xs );
    CHECK_BATCH_SIZE( grad_dirac_ws );
    CHECK_BATCH_SIZE( grad_point_xs );
    CHECK_BATCH_SIZE( grad_point_ys );

    P( grad_distances.data()[ 0 ] );
    P( grad_distances.data()[ 1 ] );

    // helpers
    const auto cview2 = [&]( const NA &v ) { return TensorView<const TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };
    const auto cview1 = [&]( const NA &v ) { return TensorView<const TF,1>{ v.data(), batch_size }; };
    const auto view2 = [&]( NA &v ) { return TensorView<TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };

    Affine1d<TF,1> grad_functions{ .xs = view2( grad_point_xs ), .ys = view2( grad_point_ys ) };
    DiracSet<TF,1> grad_diracs{ .xs = view2( grad_dirac_xs ), .ws = view2( grad_dirac_ws ) };
    Affine1d<const TF,1> functions{ .xs = cview2( point_xs ), .ys = cview2( point_ys ) };
    DiracSet<const TF,1> diracs{ .xs = cview2( dirac_xs ), .ws = cview2( dirac_ws ) };
    w2_distance_backward( cview1( grad_distances ), cview2( grad_barycenters ), cview2( barycenters ), diracs, functions, grad_diracs, grad_functions );
}

NB_MODULE( sdot_bindings_cpu, m ) {
    m.def( "ot_plan_to_piecewise_affine_1d", &ot_plan_to_piecewise_affine_1d,
          nb::arg( "dirac_xs" ), nb::arg( "dirac_ws" ),
          nb::arg( "point_xs" ), nb::arg( "point_ys" ),
          nb::arg( "result" ), nb::arg( "barycenters" ),
          "SDOT plan to get distance and barycenters with f = sum of diracs (1D), g = piecewise affine function"
    );

    m.def( "backward_ot_plan_to_piecewise_affine_1d", &backward_ot_plan_to_piecewise_affine_1d,
          nb::arg( "grad_distance" ), nb::arg( "grad_barycenters" ),
          nb::arg( "w2_barycenters" ),
          nb::arg( "dirac_xs" ), nb::arg( "dirac_ws" ),
          nb::arg( "points_xs" ), nb::arg( "points_ys" ),
          nb::arg( "grad_dirac_xs" ), nb::arg( "grad_dirac_ws" ),
          nb::arg( "grad_points_xs" ), nb::arg("grad_points_ys"),
          "SDOT W2 backward implementation"
    );
}
