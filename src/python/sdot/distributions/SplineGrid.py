from sdot.object_with_tensors._methods import TensorField, ListOfTensorFields, object_with_tensors, Distribution, driver
# from itertools import product as iproduct
from .PolynomialGrid import PolynomialGrid
from typing import TYPE_CHECKING
# import numpy as np


@object_with_tensors
class SplineGrid( Distribution ):
    """
    values      : tensor[ x, y, z, ... ]
    frame       : tensor[ nb_vec, dim ] — canonical frame by default
    knots       : [ tensor[ num_point ] for axis ] — linspace(0,1) by default
    continuity  : int — minimum continuity order (default 0 = C0)

    continuity=0  →  Q_1 bilinear per cell, local Vandermonde solve
    continuity=1  →  natural cubic spline (C2), tensor-product 1D solve per axis
    """

    values      = TensorField( "shape * dim" )
    frame       = TensorField( "dim + 1", "dim" )
    knots       = ListOfTensorFields( "dim", [ "shape[ index ]" ] )
    continuity  = 0

    if TYPE_CHECKING:
        def __init__( self, values = None, frame = None, knots = None, continuity = 0 ): ...
        shape : list[ int ]

    # polynomial order implied by the continuity requirement:
    # C0 → Q_1 (degree 1),  C1 → cubic (degree 3),  Ck → degree 2k+1
    @property
    def polynomial_order( self ):
        return 2 * self.continuity + 1

    def normalized_version( self ):
        if self.values is None:
            return PolynomialGrid( None, self.frame, self.knots )

        batch_version = 0

        dim = self.values.ndim - batch_version
        ct_dim = dim if dim <= 4 else -1

        mod = driver.import_bindings(
            f"poly_coeff_of_normalized_version_of_SplineGrid_c{ self.continuity }_{ ct_dim }d",
            lambda : _src_poly_coeff_of_normalized_version( ct_dim, self.continuity )
        )

        poly_coeffs_shapes = [ self.values.shape[ batch_version + d ] - 1 for d in range( dim ) ] + \
            [ PolynomialGrid.nb_coeffs_for( self.values.ndim, self.polynomial_order ) ]

        poly_coeffs, = driver.forward( mod.forward, mod.backward, [ poly_coeffs_shapes ], self.values, self.frame, self.knots or [] )

        return PolynomialGrid( poly_coeffs, self.frame, self.knots )


def _src_poly_coeff_of_normalized_version( ct_dim, continuity ):
    """ C++ binding for C0 spline normalization.
        1D: solves [[1,k[i]],[1,k[i+1]]] @ [c0,c1] = [v[i],v[i+1]] analytically.
        nD: for each cell, builds the 2^dim x 2^dim Vandermonde and solves by Gaussian elimination.
    """

    values_view = f"tensor_view_{ ct_dim }"
    poly_coeffs_view = f"tensor_view_{ ct_dim + 1 }"

    code = """
        #include <sdot/spline_grid_to_polynomial_coeffs.h>
        #include <sdot/nanobind_wrappers.h>
        #include <nanobind/stl/vector.h>

        namespace nb = nanobind;
        using namespace sdot;
        using TF = SDOT_SCALAR_TYPE;

        using AF = std::optional<nb::ndarray<const TF,SDOT_NANOBIND_ARCH>>;
        using MF = std::optional<nb::ndarray<TF,SDOT_NANOBIND_ARCH>>;

        NB_MODULE( SDOT_BINDING_NAME, m ) {
            m.def( "forward", []( MF _poly_coeffs, AF _values, AF _frame, const std::vector<AF> &_knots ) {
                spline_grid_to_polynomial_coeffs_forward( CtInt<BATCH_VERSION>(), CtInt<CONTINUITY>(),
                    POLY_COEFFS_VIEW( _poly_coeffs ),
                    VALUES_VIEW( _values ),
                    tensor_view_2( _frame ),
                    tensor_views_1( _knots )
                );
            } );

            m.def( "backward", []( MF _grad_values, MF _grad_frame, const std::vector<MF> &_grad_knots, AF _poly_coeffs, AF _grad_poly_coeffs, AF _values, AF _frame, const std::vector<AF> &_knots ) {
                spline_grid_to_polynomial_coeffs_backward( CtInt<BATCH_VERSION>(), CtInt<CONTINUITY>(),
                    VALUES_VIEW( _grad_values ),
                    tensor_view_2( _grad_frame ),
                    tensor_views_1( _grad_knots ),
                    POLY_COEFFS_VIEW( _poly_coeffs ),
                    POLY_COEFFS_VIEW( _grad_poly_coeffs ),
                    VALUES_VIEW( _values ),
                    tensor_view_2( _frame ),
                    tensor_views_1( _knots )
                );
            } );
        }
    """

    return driver.cpp_src( {
        "POLY_COEFFS_VIEW" : poly_coeffs_view,
        "VALUES_VIEW"      : values_view,
        "CONTINUITY"       : continuity,
        "BATCH_VERSION"    : "0",
    }, code )
