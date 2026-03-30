#pragma once

#include "MapOfUniqueSortedIndices.h"

namespace sdot {

    /**
*/
template<int ct_dim,class PII=PI32,class PIO=PI32>
class RecursiveMapOfUniqueSortedIndices {
public:
    /**/   RecursiveMapOfUniqueSortedIndices( PI dim ) : map_vec( dim ) {}

    void   prepare_for( PII max_PI_value ) { for( auto &m : map_vec ) m.prepare_for( max_PI_value ); }

    PIO&   operator[] ( const auto &key ) { return map_vec[ key.size() ][ key ]; }

private:
    using  MapVec     = std::vector<MapOfUniqueSortedIndices<-1,PII,PIO>>;

    MapVec map_vec;   ///<
};


} // namespace sdot
