#pragma once

#include "MapOfUniqueSortedIndices.h"

namespace sdot {

/** version for known ct_dim
*/
template<class InputInt,int ct_dim,class Arch>
class RecursiveMapOfUniqueSortedIndices {
public:
    /**/      RecursiveMapOfUniqueSortedIndices( PI dim ) : next_maps( dim ), curr_map( dim ) {}

    void      prepare_for                      ( InputInt max_input_value, PI64 max_output_value ) { next_maps.prepare_for( max_input_value, max_output_value ); curr_map.prepare_for( max_input_value, max_output_value ); }

    T_d auto  operator[]                       ( const DsVec<InputInt,d,Arch> &key ) { if constexpr ( d == ct_dim ) return curr_map[ key ]; else return next_maps[ key ]; }

private:
    using     NextMaps                         = RecursiveMapOfUniqueSortedIndices<InputInt,ct_dim-1,Arch>;
    using     CurrMap                          = MapOfUniqueSortedIndices<InputInt,ct_dim,Arch>;

    NextMaps  next_maps;                       ///<
    CurrMap   curr_map;                        ///<
};

/** version for ct_dim == 0
*/
template<class InputInt,class Arch>
class RecursiveMapOfUniqueSortedIndices<InputInt,0,Arch> {
public:
    /**/     RecursiveMapOfUniqueSortedIndices( PI dim ) : curr_map( dim ) {}

    void     prepare_for                      ( InputInt max_input_value, PI64 max_output_value ) { curr_map.prepare_for( max_input_value, max_output_value );  }

    auto     operator[]                       ( const DsVec<InputInt,0,Arch> &key ) { return curr_map[ key ]; }

private:
    using    CurrMap                          = MapOfUniqueSortedIndices<InputInt,0,Arch>;

    CurrMap  curr_map;                        ///<
};

/** version for unknown ct dim
*/
template<class InputInt,class Arch>
class RecursiveMapOfUniqueSortedIndices<InputInt,-1,Arch> {
public:
    /**/   RecursiveMapOfUniqueSortedIndices( PI dim ) : map_vec( dim ) {}

    void   prepare_for                      ( InputInt max_input_value, PI max_output_value ) { for( auto &m : map_vec ) m.prepare_for( max_input_value, max_output_value );  }

    auto   operator[]                       ( const auto &key ) { return map_vec[ key.size() ][ key ]; }

private:
    using  MapVec                           = std::vector<MapOfUniqueSortedIndices<InputInt,-1,Arch>>;

    MapVec map_vec;                         ///<
};


} // namespace sdot
