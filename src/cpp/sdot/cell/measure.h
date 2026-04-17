#pragma once

#include "../support/SimpleSquareMatrix.h"
#include "../support/common_types.h"
#include "for_each_simplex.h"
#include "Cell.h"
#include <limits>
#include <cmath>

namespace sdot {

/// Compute the d-dimensional measure (volume) of ``cell`` and write it into ``res()``.
///
/// Returns +∞ when the cell is not fully closed.
///
/// The measure is computed as the sum of |det| / d! over all simplices of the
/// fan triangulation.  The 2D case is handled with the shoelace formula.
template<class TF, int ct_dim, class Arch>
void measure( auto &res, const Cell<TF,ct_dim,Arch> &cell ) {
    const PI nb_vertices = cell.nb_vertices();
    const PI dim = cell.dim();

    // infinite cell
    if ( ! cell.is_fully_closed() ) {
        res() = std::numeric_limits<TF>::infinity();
        return;
    }

    // 2D: shoelace formula
    if ( dim == 2 ) {
        TF sum = 0;
        for ( PI i = 0; i < nb_vertices; ++i ) {
            const PI j = ( i + 1 ) % nb_vertices;
            sum += cell.vertex_positions( i, 0 ) * cell.vertex_positions( j, 1 )
                 - cell.vertex_positions( j, 0 ) * cell.vertex_positions( i, 1 );
        }
        res() = sum / 2;
        return;
    }

    // nD: fan triangulation
    SimpleSquareMatrix<TF,ct_dim,Arch> M( dim );
    TF sum = 0;

    for_each_simplex( cell, [&]( const auto &simplex ) {
        const PI v0 = simplex[ 0 ];
        auto M = SimpleSquareMatrix<TF,ct_dim,Arch>::with_func( dim, [&]( PI row, PI col ) {
            return cell.vertex_positions( simplex[ col + 1 ], row ) - cell.vertex_positions( v0, row );
        } );
        sum += std::abs( M.determinant() );
    } );

    res() = sum / factorial( dim );
}

} // namespace sdot
