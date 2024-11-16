#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/compare.h>
#include <unordered_map>

struct DiracVecHash { 
    PI operator()( const auto &vr ) const {
        PI m = 1, r = 0;
        for( auto v : vr ) {
            r += m * v;
            m *= nb_refs;
        }
        return r;
    }
    PI nb_refs;
};

template<int dim,class nb_dims_> struct RefMapForDim {
    static constexpr int nb_dims = nb_dims_::value;

    RefMapForDim( PI nb_refs ) : map( nb_refs, DiracVecHash( nb_refs ) ) {
    }

    PI find( const Vec<PI,dim> vr, auto &&on_new_item ) {
        auto iter = map.find( vr );
        if ( iter == map.end() ) {
            iter = map.insert( iter, { vr, refs.size() } );
            refs << vr;

            on_new_item( iter->second );
        }
        return iter->second;
    }

    void on_new_ref() {
        for( Vec<Vec<PI>> &p : parenting )
            p.push_back();
    }

    std::unordered_map<Vec<PI,dim>,PI,DiracVecHash,Equal> map;
    Vec<Vec<Vec<PI>>,nb_dims+1> parenting; ///< [ parent dim ][ num_item ] => [ parent items ]
    Vec<Vec<PI,dim>> refs;
};

template<class nb_dims_> struct RefMapForDim<1,nb_dims_> {
    static constexpr int nb_dims = nb_dims_::value;

    RefMapForDim( PI nb_refs ) : seen( FromSizeAndItemValue(), nb_refs, false ) {
        for( Vec<Vec<PI>> &p : parenting )
            p.resize( nb_refs );
    }

    PI find( const Vec<PI,1> vr, auto &&on_new_item ) {
        if ( seen[ vr[ 1 ] ] == false ) {
            seen[ vr[ 1 ] ] = true;
            on_new_item( vr[ 1 ] );
        }
        return vr[ 1 ];
    }

    void on_new_ref() {
    }

    Vec<Vec<Vec<PI>>,nb_dims+1> parenting; ///< [ parent dim ][ num_item ] => [ parent items ]
    Vec<bool> seen;
};

template<int d,int nb_dims>
void get_rec_item_data( auto &vertex_coords, const auto &vertex_coord, auto &ref_map, Vec<PI,d> refs, CtInt<nb_dims>, const auto &children_indices ) {
    if constexpr ( d ) {
        auto &rm = ref_map[ refs.size() ];

        // if this item has not yet been seen
        PI ind = rm.find( refs, /*on new item*/ [&]( PI new_ind ) {
            // if on a vertex, add the coordinates 
            if ( d == nb_dims + 1 )
                vertex_coords << vertex_coord;

            // register new parenting vectors
            rm.on_new_ref();

            // recursion
            for( PI i = 0; i < d; ++i )
                get_rec_item_data( vertex_coords, vertex_coord, ref_map, refs.without_index( i ), CtInt<nb_dims>(), children_indices.with_pushed_value( new_ind ) );
        } );

        //
        for( PI e = 0; e < children_indices.size(); ++e ) {
            rm.parenting[ e ][ ind ] << children_indices[ e ];
        }
    }
}
