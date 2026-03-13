#pragma once

#include "Affine1d.h"

namespace sdot {

    // --------------------------------------------------- batch dim 0 ---------------------------------------------------
#define UTP template<class T>
#define DTP Affine1d<T,0>

UTP DTP::Piece DTP::get_first_piece() const {
    if ( nb_points() < 2 )
        throw std::runtime_error( "For affine function, we need at least 2 points" );

    const TF x0 = static_cast<TF>( xs[ 0 ] );
    const TF x1 = static_cast<TF>( xs[ 1 ] );
    const TF y0 = static_cast<TF>( ys[ 0 ] );
    const TF y1 = static_cast<TF>( ys[ 1 ] );

    const TF mass = ( x1 - x0 ) * ( y1 + y0 ) / 2.0;

    return {
        .index = 1,
        .x0    = x0,
        .x1    = x1,
        .y0    = y0,
        .y1    = y1,
        .mass  = mass
    };
}

UTP void DTP::get_next_piece( Piece &piece ) const {
    if ( ++piece.index < nb_points() ) {
        piece.x0 = std::exchange( piece.x1, static_cast<TF>( xs[ piece.index ] ) );
        piece.y0 = std::exchange( piece.y1, static_cast<TF>( ys[ piece.index ] ) );
        piece.mass = ( piece.x1 - piece.x0 ) * ( piece.y1 + piece.y0 ) / 2;
        return;
    }

    piece.x0 = piece.x1;
    piece.y0 = piece.y1;
    piece.mass = std::numeric_limits<TF>::max();
}

UTP void DTP::take_some_mass( Piece &current_piece, TF mass_to_take, auto &&on_taken_piece ) const {
    // enough mass in the current piece ?
    if ( mass_to_take <= current_piece.mass ) {
        Piece np = current_piece.take_some_mass( mass_to_take );
        on_taken_piece( np );
        return;
    }

    // else, use the current piece, get to the next, ...
    mass_to_take -= current_piece.mass;
    on_taken_piece( current_piece );

    while ( true ) {
        get_next_piece( current_piece );
        if ( mass_to_take <= current_piece.mass )
           break;

        // full interval
        mass_to_take -= current_piece.mass;
        on_taken_piece( current_piece );
    }

    Piece np = current_piece.take_some_mass( mass_to_take );
    on_taken_piece( np );
}

UTP PI DTP::nb_points() const {
    return xs.size();
}

UTP DTP::TF DTP::mass() const {
    TF px0 = static_cast<TF>( xs[ 0 ] );
    TF py0 = static_cast<TF>( ys[ 0 ] );
    TF res = 0;
    for( PI j = 1; j < nb_points(); ++j ) {
        const TF px1 = static_cast<TF>( xs[ j ] );
        const TF py1 = static_cast<TF>( ys[ j ] );
        res += ( py0 + py1 ) * ( px1 - px0 );
        px0 = px1;
        py0 = py1;
    }
    return res / 2;
}

#undef UTP
#undef DTP

// --------------------------------------------------- batch dim 1 ---------------------------------------------------
#define UTP template<class T>
#define DTP Affine1d<T,1>

UTP PI DTP::nb_rows() const {
    return xs.size( 0 );
}

UTP Affine1d<T,0> DTP::row( PI num_batch ) const  {
   return { xs.row( num_batch ), ys.row( num_batch ) };
}

#undef UTP
#undef DTP


} // namespace sdot
