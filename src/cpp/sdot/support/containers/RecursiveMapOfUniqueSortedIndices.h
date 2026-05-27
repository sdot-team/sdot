#pragma once

#include "MapOfUniqueSortedIndices.h"
#include "TensorView.h"

namespace sdot {

/** version for known ct_dim
*/
template<int ct_dim,class TV>
class RecursiveMapOfUniqueSortedIndices {
public:
    using Next = RecursiveMapOfUniqueSortedIndices<ct_dim-1,TV>;
    using Curr = MapOfUniqueSortedIndices<ct_dim,TV>;
    using TI = TV::value_type;

    HD RecursiveMapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, TI max_inp_value ) :
        curr( map_items, ( nb_map_items = 0 ), max_inp_value ),
        next( map_items, nb_map_items, max_inp_value ) {
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
template<class TV>
class RecursiveMapOfUniqueSortedIndices<0,TV> {
public:
    using Curr = MapOfUniqueSortedIndices<0,TV>;
    using TI = TV::value_type;

    HD RecursiveMapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, TI max_inp_value ) : curr( map_items, nb_map_items, max_inp_value ) {
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

template<class T,T dim>
HD auto recursive_map_of_unique_sorted_indices( Ct<T,dim>, auto &&map_items, auto &&nb_map_items, PI max_inp_value ) {
    return RecursiveMapOfUniqueSortedIndices<dim,DECAYED_TYPE_OF( map_items )>( map_items, nb_map_items, max_inp_value );
}


} // namespace sdot
