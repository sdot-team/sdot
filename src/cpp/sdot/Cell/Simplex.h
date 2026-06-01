#pragma once

#include "../support/containers/Matrix.h"
#include "../support/containers/Vector.h"
#include "../support/TODO.h"

namespace sdot {

template<int dim,int npt,class TF>
struct Simplex {
    using Pt = Vector<TF,dim>;

    TF measure() const {
        using namespace std;
        if constexpr ( npt == dim + 1 ) {
            auto M = Matrix<TF,dim>::with_func( [&]( auto row, auto col ) { return pts[ row + 1 ][ col ] - pts[ 0 ][ col ]; } );
            return abs( M.determinant() );
        } else {
            TODO;
        }
    }

    Pt centroid() const {
        Pt res = pts[ 0 ];
        for( PI i = 1; i < npt; ++i )
            res += pts[ i ];
        return res / npt;
    }

    std::array<Pt,npt> pts;
};

} // namespace sdot
