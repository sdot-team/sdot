#include "../../src/cpu/w2_distance.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace sdot;
using TF = float;

// using ndarray_f32_2d = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
// using ndarray_f32 = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NA = nb::ndarray<TF>;

// Forward function that works with both 1D and 2D arrays
void sdot_w2_forward_impl( NA dirac_xs, NA dirac_ws, NA point_xs, NA point_ys, NA distance, NA barycenters ) {
    // batch_size
    if ( dirac_xs.ndim() != dirac_ws.ndim() || dirac_xs.ndim() != point_xs.ndim() || dirac_xs.ndim() != point_ys.ndim() )
        throw std::invalid_argument( "input tensors must have same rank" );
    const PI batch_size = dirac_xs.ndim() == 2 ? dirac_xs.shape( 0 ) : 1;

    // helpers
    const auto cview2 = [&]( const NA &v ) { return TensorView<const TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };
    // const auto cview1 = [&]( const NA &v ) { return TensorView<const TF,1>{ v.data(), batch_size }; };
    const auto view2 = [&]( NA &v ) { return TensorView<TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };
    const auto view1 = [&]( NA &v ) { return TensorView<TF,1>{ v.data(), batch_size }; };

    Affine1d<const TF,1> functions{ .xs = cview2( point_xs ), .ys = cview2( point_ys ) };
    DiracSet<const TF,1> diracs{ .xs = cview2( dirac_xs ), .ws = cview2( dirac_ws ) };
    w2_distance( diracs, functions, view1( distance ), view2( barycenters ) );
}

void sdot_w2_backward_impl(
    NA grad_distance, NA grad_barycenters, NA barycenters,
    NA dirac_xs, NA dirac_ws, NA point_xs, NA point_ys,
    NA grad_dirac_xs, NA grad_dirac_ws, NA grad_point_xs, NA grad_point_ys
) {
    // batch_size
    if ( dirac_xs.ndim() != dirac_ws.ndim() || dirac_xs.ndim() != point_xs.ndim() || dirac_xs.ndim() != point_ys.ndim() )
        throw std::invalid_argument( "input tensors must have same rank" );
    const PI batch_size = dirac_xs.ndim() == 2 ? dirac_xs.shape( 0 ) : 1;

    // helpers
    const auto cview2 = [&]( const NA &v ) { return TensorView<const TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };
    const auto cview1 = [&]( const NA &v ) { return TensorView<const TF,1>{ v.data(), batch_size }; };
    const auto view2 = [&]( NA &v ) { return TensorView<TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };

    // TV grad_bary( nb_diracs * batch_size, 0 );
    // TV grad_dist( batch_size, 1 );
    Affine1d<TF,1> grad_functions{ .xs = view2( grad_point_xs ), .ys = view2( grad_point_ys ) };
    DiracSet<TF,1> grad_diracs{ .xs = view2( grad_dirac_xs ), .ws = view2( grad_dirac_ws ) };
    Affine1d<const TF,1> functions{ .xs = cview2( point_xs ), .ys = cview2( point_ys ) };
    DiracSet<const TF,1> diracs{ .xs = cview2( dirac_xs ), .ws = cview2( dirac_ws ) };
    w2_distance_backward( cview1( grad_distance ), cview2( grad_barycenters ), cview2( barycenters ), diracs, functions, grad_diracs, grad_functions );
}

NB_MODULE( sdot_pytorch_bindings, m ) {
    m.def( "forward_impl", &sdot_w2_forward_impl,
          nb::arg( "dirac_xs" ), nb::arg( "dirac_ws" ),
          nb::arg( "point_xs" ), nb::arg( "point_ys" ),
          nb::arg( "result" ), nb::arg( "barycenters" ),
          "SDOT W2 forward implementation"
    );

    m.def( "backward_impl", &sdot_w2_backward_impl,
          nb::arg( "grad_distance" ), nb::arg( "grad_barycenters" ),
          nb::arg( "w2_barycenters" ),
          nb::arg( "dirac_xs" ), nb::arg( "dirac_ws" ),
          nb::arg( "points_xs" ), nb::arg( "points_ys" ),
          nb::arg( "grad_dirac_xs" ), nb::arg( "grad_dirac_ws" ),
          nb::arg( "grad_points_xs" ), nb::arg("grad_points_ys"),
          "SDOT W2 backward implementation"
    );
}
