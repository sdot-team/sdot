#line 39 "/Users/hugo.leclerc/sdot_with_interfaces/src/python/sdot/plan.py"
#include "../../../../../cpp/cpu/w2_distance.h"
#include "../../nanobind_wrappers.h"

namespace nb = nanobind;
using namespace sdot;

using NA = nanobind::device::cpu;
using TF = FP64;

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

MK_MOD( ot_plan_BatchOfSumOfWeightedDiracs1d_BatchOfPiecewiseAffineFunction1d_1d_FP64_cpu ) {
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
