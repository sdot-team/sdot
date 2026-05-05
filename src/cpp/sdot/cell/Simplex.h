#pragma once

#include "../support/SimpleSquareMatrix.h"
#include "../support/DsVec.h"
#include "../support/TODO.h"

namespace sdot {

template<int dim,int npt,class TF,class Arch>
struct Simplex {
    using Pt = DsVec<TF,dim,Arch>;

    TF measure() const {
        using namespace std;
        if constexpr ( npt == dim + 1 ) {
            auto M = SimpleSquareMatrix<TF,dim,Arch>::with_func( dim, [&]( auto row, auto col ) { return pts[ row + 1 ][ col ] - pts[ 0 ][ col ]; } );
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
