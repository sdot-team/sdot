#pragma once

#include "MapOfUniqueSortedIndices.h"
#include "TensorView.h"

namespace sdot {

/** version for known ct_dim
*/
template<int ct_dim,class TI,class Arch>
class RecursiveMapOfUniqueSortedIndices {
public:
    using Next = RecursiveMapOfUniqueSortedIndices<ct_dim-1,TI,Arch>;
    using Curr = MapOfUniqueSortedIndices<ct_dim,TI,Arch>;
    using TV = TensorView<TI,AxisTuple<TI,Arch,1>,AxisTuple<TI,Arch,1>>;

    /**/ RecursiveMapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, int dim, TI max_inp_value ) :
        curr( map_items, ( nb_map_items = 0 ), dim, max_inp_value ),
        next( map_items, nb_map_items, dim, max_inp_value ) {
    }

    void reserve_full_capacity() {
        curr.reserve_full_capacity();
        next.reserve_full_capacity();
    }

    void reserve( TI reservation ) {
        curr.reserve( reservation );
        next.reserve( reservation );
    }

    template<int i>
    IntWithOffset<TI> operator[]( const Vector<TI,Arch,i> &key ) {
        if constexpr( i == ct_dim )
            return curr[ key ];
        else
            return next[ key ];
    }

    Curr curr;
    Next next;
};

/** version for ct_dim == 0
*/
template<class TI,class Arch>
class RecursiveMapOfUniqueSortedIndices<0,TI,Arch> {
public:
    using Curr = MapOfUniqueSortedIndices<0,TI,Arch>;
    using TV = TensorView<TI,AxisTuple<TI,Arch,1>,AxisTuple<TI,Arch,1>>;

    /**/ RecursiveMapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, int dim, TI max_inp_value ) : curr( map_items, nb_map_items, dim, max_inp_value ) {
    }

    void reserve_full_capacity() {
        curr.reserve_full_capacity();
    }

    void reserve( TI reservation ) {
        curr.reserve( reservation );
    }

    IntWithOffset<TI> operator[]( const Vector<TI,Arch,0> &key ) {
        return curr.operator[]( key );
    }

    Curr curr;
};

// /** version for unknown ct dim
// */
// template<class InputInt,class Arch>
// class RecursiveMapOfUniqueSortedIndices<InputInt,-1,Arch> {
// public:
//     /**/   RecursiveMapOfUniqueSortedIndices( PI dim ) : map_vec( dim ) {}

//     void   prepare_for                      ( InputInt max_input_value, PI max_output_value ) { for( auto &m : map_vec ) m.prepare_for( max_input_value, max_output_value );  }

//     auto   operator[]                       ( const auto &key ) { return map_vec[ key.size() ][ key ]; }

// private:
//     using  MapVec                           = std::vector<MapOfUniqueSortedIndices<InputInt,-1,Arch>>;

//     MapVec map_vec;                         ///<
// };


} // namespace sdot
