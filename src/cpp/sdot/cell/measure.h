#pragma once

#include "../support/SimpleSquareMatrix.h"
#include "for_each_simplex.h"
#include "CellWorker.h"
#include <limits>
#include <cmath>

namespace sdot {

/// Compute the d-dimensional measure (volume) of ``cell`` and write it into ``res()``.
///
/// Returns +∞ when the cell is not fully closed.
///
/// The measure is computed as the sum of |det| / d! over all simplices of the
/// fan triangulation.  The 2D case is handled with the shoelace formula.
void measure( auto &&p ) {
    TODO;
}

template<int ct_dim,class Arch,class TF,class TI>
void measure_backward( const auto &/*res_measure*/, const Cell<ct_dim,Arch,TF,TI> &cell, auto &grad_vertex_positions, auto &/*grad_cut_planes*/, const auto &grad_out_measure ) {
    const PI nb_vertices = cell.nb_vertices();
    const PI dim = cell.dim();

    if ( dim == 2 ) {
        const TF g = grad_out_measure() / 2;
        for ( PI k = 0; k < nb_vertices; ++k ) {
            const PI kp = ( k + 1 ) % nb_vertices;
            const PI km = ( k + nb_vertices - 1 ) % nb_vertices;
            grad_vertex_positions( k, 0 ) += g * ( cell.vertex_positions( kp, 1 ) - cell.vertex_positions( km, 1 ) );
            grad_vertex_positions( k, 1 ) += g * ( cell.vertex_positions( km, 0 ) - cell.vertex_positions( kp, 0 ) );
        }
        return;
    }

    TODO; // nD: gradient via fan triangulation
}

} // namespace sdot
