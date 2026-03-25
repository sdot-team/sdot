#pragma once

#include "support/TODO.h"
#include "DiracSet.h"
#include <algorithm>

namespace sdot {

#define UTP template<class T,class Arch>
#define DTP DiracSet<T,Arch>

UTP PI DTP::nb_diracs() const {
    return xs.size();
}

UTP DTP::TF DTP::mass() const {
    TF res = 0;
    for( PI i = 0; i < nb_diracs(); ++i )
        res += static_cast<TF>( ws[ i ] );
    return res;
}

UTP auto DTP::arg_sort() -> std::vector<PI> const {
    std::vector<PI> res( nb_diracs() );

    for( PI i = 0; i < nb_diracs(); ++i )
        res[ i ] = i;

    std::sort( res.begin(), res.end(), [&]( PI index_a, PI index_b ) {
        return xs[ index_a ] < xs[ index_b ];
    });

    return res;
}

#undef UTP
#undef DTP

#define UTP template<class T,class Arch>
#define DTP BatchOfDiracSet<T,Arch>

UTP PI DTP::nb_rows() const {
    return xs.size( 0 );
}

UTP DiracSet<T,Arch> DTP::row( PI num_batch ) const  {
   return { xs.row( num_batch ), ws.row( num_batch ) };
}

UTP auto DTP::masses() const -> Tensor<TF,1,Arch> {
    return ws.template sum_axis_0<TF>();
}

#undef UTP
#undef DTP

} // namespace sdot
