#pragma once

#include <tl/support/operators/product.h>
#include <tl/support/operators/norm_2.h>

#include "VoronoiAccelerationStructure.h"
#include "CellTraversalError.h"

#include "../support/spawn.h"
#include "../support/Mpi.h"

#include "../CutType.h"

#include <tl/support/P.h>
#include <cmath>

namespace sdot {

#define DTP template<class TCell>
#define UTP VoronoiAccelerationStructure<TCell>

DTP UTP::VoronoiAccelerationStructure( const PoomVec<Pt> &positions, const PoomVec<TF> &weights, const Vec<Trans> &transformations, TF nb_diracs_per_box ) : transformations( transformations ) {
    // using std::pow;

    // // local min_pos, max_pos + nb_base_points
    // bool seen = false;
    // nb_glob_points = 0;
    // nb_base_points = 0;
    // point_reader.get_local_content( [&]( const auto &content ) {
    //     if ( content.size() == 0 )
    //         return;

    //     if ( ! seen ) {
    //         limits[ 0 ] = *content.begin();
    //         limits[ 1 ] = limits[ 0 ];
    //         seen = true;
    //     }

    //     for( const auto &pt : content ) {
    //         limits[ 0 ] = min( limits[ 0 ], pt );
    //         limits[ 1 ] = max( limits[ 1 ], pt );
    //         for( const auto &trans : transformations ) {
    //             limits[ 0 ] = min( limits[ 0 ], trans( pt ) );
    //             limits[ 1 ] = max( limits[ 1 ], trans( pt ) );
    //         }
    //     }
        
    //     nb_glob_points += content.size() * ( 1 + transformations.size() );
    //     nb_base_points += content.size();
    // } );

    // // global min and max pos + nb points
    // limits = mpi->reduction( limits, []( Vec<Pt,2> &a, const Vec<Pt,2> &b ) {
    //     a[ 0 ] = min( a[ 0 ], b[ 0 ] );
    //     a[ 1 ] = max( a[ 1 ], b[ 1 ] );
    // }, {} );

    // nb_glob_points = mpi->sum( nb_glob_points );
    // nb_base_points = mpi->sum( nb_base_points );

    // // divs
    // TF step = pow( nb_diracs_per_box / ( nb_glob_points * product( limits[ 1 ] - limits[ 0 ] ) ), double( 1 ) / nb_dims );
    // for( PI d = 0; d < nb_dims; ++d ) {
    //     nb_divs[ d ] = max( TF( 1 ), ceil( ( limits[ 1 ][ d ] - limits[ 0 ][ d ] ) / step ) );
    //     steps[ d ] = TF( limits[ 1 ][ d ] - limits[ 0 ][ d ] ) / nb_divs[ d ];
    // }

    // // get local nb diracs for each cell (stored in this->offsets)
    // offsets.resize( 2 * product( nb_divs ) + 1, 0 );
    // point_reader.get_local_content( [&]( const auto &content ) {
    //     for( const auto &pt : content ) {
    //         ++offsets[ 2 * index( pt ) + 0 ];
    //         for( const auto &trans : transformations )
    //             ++offsets[ 2 * index( trans( pt ) ) + 1 ];
    //     }
    // } );

    // // exclusive scan
    // if ( mpi->size() > 1 )
    //     TODO; // make a repartition, gather offsets
    // for( PI i = 0, a = 0; i < offsets.size(); ++i )
    //     offsets[ i ] = std::exchange( a, a + offsets[ i ] );

    // // get points and inds
    // points.resize( nb_glob_points );
    // inds.resize( nb_glob_points );
    // PI ind = 0;
    // point_reader.get_local_content( [&]( const auto &content ) {
    //     // base points
    //     for( const auto &pt : content ) {
    //         PI o = offsets[ 2 * index( pt ) + 0 ]++;
    //         points[ o ] = pt;
    //         inds[ o ] = ind;

    //         for( const auto &trans : transformations ) {
    //             PI o = offsets[ 2 * index( trans( pt ) ) + 1 ]++;
    //             points[ o ] = trans( pt );
    //             inds[ o ] = ind;
    //         }

    //         ++ind;
    //     }
    // } );

    // // remake the exclusive scan
    // for( PI i = 0, o = 0; i < offsets.size() - 1; ++i )
    //     offsets[ i ] = std::exchange( o, offsets[ i ] );
}

DTP void UTP::display( Displayer &ds ) const {
    // DS_OBJECT( limits, nb_divs );
    ds << "pouet";
}

DTP PI UTP::recommended_nb_threads() const {
    return std::thread::hardware_concurrency();
}

// DTP void UTP::make_cuts_from( PI b0, PI n0, TCell &cell, Vec<PI> &buf ) {
//     // indices of points to be tested (includes virtual points)
//     buf.clear();
//     for( PI n1 = offsets[ 2 * b0 + 0 ]; n1 < offsets[ 2 * b0 + 2 ]; ++n1 )
//         if ( n1 != n0 )
//             buf << n1;

//     // sort indices
//     std::sort( buf.begin(), buf.end(), [&]( PI a, PI b ) -> bool {
//         return norm_2_p2( points[ a ] - cell.info.p0 ) < norm_2_p2( points[ b ] - cell.info.p0 );
//     } );

//     // make the cuts
//     const Pt p0 = cell.info.p0;
//     const TF w0 = cell.info.w0;
//     for( PI n1 : buf ) {
//         const Pt p1 = points[ n1 ];
//         const PI i1 = inds[ n1 ];
//         const TF w1 = 1;

//         const Pt dir = p1 - p0;

//         auto n = norm_2_p2( dir );
//         auto s0 = sp( dir, p0 );
//         auto s1 = sp( dir, p1 );

//         auto off = s0 + ( 1 + ( w0 - w1 ) / n ) / 2 * ( s1 - s0 );

//         cell.cut( dir, off, { CutType::Dirac, p1, w1, i1 } );
//     }
// }

DTP int UTP::for_each_cell( const TCell &base_cell, auto &&f, int max_nb_threads ) {
    // // for each box...
    // int error = 0;
    // spawn( [&]( int num_thread, int nb_threads ) {
    //     try {
    //         PI beg_b0 = end_index() * ( num_thread + 0 ) / nb_threads;
    //         PI end_b0 = end_index() * ( num_thread + 1 ) / nb_threads;

    //         TCell local_cell;
    //         Vec<PI> buf( FromReservationSize(), 128 );
    //         for( PI b0 = beg_b0; b0 < end_b0; ++b0 ) {
    //             // "primary" points (not the virtual ones)
    //             for( PI n0 = offsets[ 2 * b0 + 0 ]; n0 < offsets[ 2 * b0 + 1 ]; ++n0 ) {
    //                 if ( error )
    //                     return;
    //                 const PI i0 = inds[ n0 ];

    //                 // copy of base cell
    //                 local_cell.get_geometrical_data_from( base_cell );
    //                 local_cell.info.w0 = 1;
    //                 local_cell.info.p0 = points[ n0 ];
    //                 local_cell.info.i0 = i0;

    //                 // cuts with points from b0
    //                 for( PI b1 = 0; b1 < end_index(); ++b1 )
    //                     make_cuts_from( b1, n0, local_cell, buf );

    //                 // callback
    //                 f( local_cell, num_thread );
    //             }
    //         }
    //     } catch ( CellTraversalError e ) {
    //         error = e.error;
    //     }
    // }, max_nb_threads ? max_nb_threads : recommended_nb_threads() );
    // return error;
    return 0;
}

#undef DTP
#undef UTP

} // namespace sdot
