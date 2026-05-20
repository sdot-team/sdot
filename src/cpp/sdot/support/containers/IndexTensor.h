#pragma once

#include "common_macros.h"
#include "common_types.h"

namespace sdot {

/// view on strided data (strides in bytes, handles non-contiguous arrays)
template<class T,int ct_rank,int index,class C=T>
class IndexTensor {
public:
    HD auto        operator()( PI i0, PI i1, PI i2 ) const { static_assert( ct_rank == 3 ); if ( index == 0 ) return coeff * i0; if ( index == 1 ) return coeff * i1; if ( index == 2 ) return coeff * i2; return coeff * T( 0 ); }
    HD auto        operator()( PI i0, PI i1 ) const { static_assert( ct_rank == 2 ); if ( index == 0 ) return coeff * i0; if ( index == 1 ) return coeff * i1; return coeff * T( 0 ); }
    HD auto        operator()( PI i0 ) const { static_assert( ct_rank == 1 ); if ( index == 0 ) return coeff * i0; return coeff * T( 0 ); }
    HD auto        operator()() const { return 0; }

    HD auto        operator[]( PI i0 ) const { static_assert( ct_rank == 1 ); if ( index == 0 ) return coeff * i0; return coeff * T( 0 ); }

    HD PI          rank      () const { return ct_rank; }

    HD auto        row       ( PI ) const { return IndexTensor<T,ct_rank-1,index-1>{ coeff }; }

    C              coeff     = 1;
};

} // namespace sdot

