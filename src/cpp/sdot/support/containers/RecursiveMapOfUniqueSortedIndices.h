#pragma once

#include "MapOfUniqueSortedIndices.h"
#include "TensorView.h"

namespace sdot {

/** version for known ct_dim
*/
template<int ct_dim,class TI,class MemorySpace>
class RecursiveMapOfUniqueSortedIndices {
public:
    using Next = RecursiveMapOfUniqueSortedIndices<ct_dim-1,TI,MemorySpace>;
    using Curr = MapOfUniqueSortedIndices<ct_dim,TI,MemorySpace>;
    using TV = TensorView<TI,MemorySpace,Tuple<TI>>;

    HD RecursiveMapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, int dim, TI max_inp_value ) :
        curr( map_items, ( nb_map_items = 0 ), dim, max_inp_value ),
        next( map_items, nb_map_items, dim, max_inp_value ) {
    }

    HD void reserve_full_capacity() {
        curr.reserve_full_capacity();
        next.reserve_full_capacity();
    }

    HD void reserve( TI reservation ) {
        curr.reserve( reservation );
        next.reserve( reservation );
    }

    template<int i>
    HD IntWithOffset<TI> operator[]( const Vector<TI,i> &key ) {
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
template<class TI,class MemorySpace>
class RecursiveMapOfUniqueSortedIndices<0,TI,MemorySpace> {
public:
    using Curr = MapOfUniqueSortedIndices<0,TI,MemorySpace>;
    using TV = TensorView<TI,MemorySpace,Tuple<TI>>;

    HD RecursiveMapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, int dim, TI max_inp_value ) : curr( map_items, nb_map_items, dim, max_inp_value ) {
    }

    HD void reserve_full_capacity() {
        curr.reserve_full_capacity();
    }

    HD void reserve( TI reservation ) {
        curr.reserve( reservation );
    }

    HD IntWithOffset<TI> operator[]( const Vector<TI,0> &key ) {
        return curr.operator[]( key );
    }

    Curr curr;
};

// /** version for unknown ct dim
// */
// template<class InputInt,class MemorySpace>
// class RecursiveMapOfUniqueSortedIndices<InputInt,-1,MemorySpace> {
// public:
//     /**/   RecursiveMapOfUniqueSortedIndices( PI dim ) : map_vec( dim ) {}

//     void   prepare_for                      ( InputInt max_input_value, PI max_output_value ) { for( auto &m : map_vec ) m.prepare_for( max_input_value, max_output_value );  }

//     auto   operator[]                       ( const auto &key ) { return map_vec[ key.size() ][ key ]; }

// private:
//     using  MapVec                           = std::vector<MapOfUniqueSortedIndices<InputInt,-1,MemorySpace>>;

//     MapVec map_vec;                         ///<
// };


} // namespace sdot
