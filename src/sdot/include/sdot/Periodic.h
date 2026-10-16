#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// A DISTRIBUTION MADE PERIODIC: the sum of its images `x -> linear( t ) x + shifts( t )`.
//
// With a periodicity, the cell of a seed is built in the plane ( unfolded, around its seed: `diagram/Common.h` ), and
// the part of it that lies beyond the support of the density is where an IMAGE of the density lives -- the cell
// meets at most the images that touch the start cell, and those are the images of the group the diagram carries ( `PowerDiagram.py`,
// `_periodic_setup` ). Each image cuts the cell into its own pieces, so the moments, the facets and the adjoint
// are those of the cell where it lies, in the frame of its seed: nothing is folded back after the fact.
//
// A distribution takes part by integrating itself MOVED by an isometry, without being copied:
//
//     for_each_piece_moved( cell, ws, func, L, s )   the pieces of `cell` against the density carried by `x -> L x + s`
//
// ( `Image`, `Polytope` ). Its parameters receive their gradient through the same closures: the images share them.
// =====================================================================================

#include <loom/support/containers/Vector.h>
#include <loom/support/containers/Matrix.h>

namespace sdot {

template<class Dist,class Shifts,class Linear>
struct PeriodicDist {
    static constexpr bool cuts_pieces = [] { if constexpr ( requires { Dist::cuts_pieces; } ) return bool( Dist::cuts_pieces ); else return false; }();

    Dist   dist;
    Shifts shifts;                                       ///< `[ nb_images, dim ]`, row 0 being zero
    Linear linear;                                       ///< `[ nb_images, dim, dim ]`, or absent ( translations )

    HD void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
        using TF = typename DECAYED_TYPE_OF( shifts )::TF;
        constexpr int D = DECAYED_TYPE_OF( cell )::ct_dim;
        static_assert( requires { dist.for_each_piece_moved( cell, ws, func, Matrix<TF,D>::identity(), Vector<TF,D>::zeros() ); },
                       "periodicity: this distribution does not know how to integrate itself moved ( `for_each_piece_moved` )" );
        const SI nb = SI( shifts.shape( 0 ) );
        for ( SI t = 0; t < nb && ! ws.overflow; ++t ) {
            const auto s = Vector<TF,D>::with_func( [&]( PI d ) { return TF( shifts( t, d ) ); } );
            if constexpr ( DECAYED_TYPE_OF( linear )::is_valid )
                dist.for_each_piece_moved( cell, ws, func, Matrix<TF,D>::with_func( [&]( auto r, auto c ) { return TF( linear( t, r, c ) ); } ), s );
            else
                dist.for_each_piece_moved( cell, ws, func, Matrix<TF,D>::identity(), s );
        }
    }
};

template<class Dist,class Shifts,class Linear>
HD auto periodic_dist( const Dist &dist, const Shifts &shifts, const Linear &linear ) {
    return PeriodicDist<Dist,Shifts,Linear>{ dist, shifts, linear };
}

} // namespace sdot
