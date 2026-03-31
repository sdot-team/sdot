#pragma once

// #include "support/TODO.h"
// #include "support/P.h"
#include "SumOfWeightedDiracs1d.h"
#include <algorithm>

namespace sdot {

#define UTP template<class T,class Arch>
#define DTP SumOfWeightedDiracs1d<T,Arch>

UTP PI DTP::nb_diracs() const {
    return positions.size();
}

UTP DTP::TF DTP::mass() const {
    TF res = 0;
    for( PI i = 0; i < nb_diracs(); ++i )
        res += static_cast<TF>( weights[ i ] );
    return res;
}

UTP auto DTP::arg_sort() -> std::vector<PI> const {
    std::vector<PI> res( nb_diracs() );

    for( PI i = 0; i < nb_diracs(); ++i )
        res[ i ] = i;

    std::sort( res.begin(), res.end(), [&]( PI index_a, PI index_b ) {
        return positions[ index_a ] < positions[ index_b ];
    });

    return res;
}

#undef UTP
#undef DTP

} // namespace sdot
