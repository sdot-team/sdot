#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/compare.h>
#include <unordered_map>
#include <algorithm>

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

    RefMapForDim( PI nb_diracs, PI nb_bnds ) : map( nb_diracs + nb_bnds, DiracVecHash( nb_diracs + nb_bnds ) ) {
    }

    PI find( const Vec<PI,dim+1> vr, auto &&on_new_item ) {
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

    std::unordered_map<Vec<PI,dim+1>,PI,DiracVecHash,Equal> map;
    Vec<Vec<Vec<PI>>,nb_dims+1> parenting; ///< [ parent dim ][ num_item ] => [ parent items ]
    Vec<Vec<PI,dim+1>> refs;
};

template<class nb_dims_> struct RefMapForDim<0,nb_dims_> {
    static constexpr int nb_dims = nb_dims_::value;

    RefMapForDim( PI nb_diracs, PI nb_bnds ) : seen( FromSizeAndItemValue(), nb_diracs, false ) {
        for( Vec<Vec<PI>> &p : parenting )
            p.resize( nb_diracs );
    }

    PI find( const Vec<PI,1> vr, auto &&on_new_item ) {
        if ( seen[ vr[ 0 ] ] == false ) {
            seen[ vr[ 0 ] ] = true;
            on_new_item( vr[ 0 ] );
        }
        return vr[ 0 ];
    }

    void on_new_ref() {
    }

    Vec<Vec<Vec<PI>>,nb_dims+1> parenting; ///< [ parent dim ][ num_item ] => [ parent items ]
    Vec<bool> seen;
};

template<int d,int nb_dims>
void get_rec_item_data( auto &vertex_coords, const auto &vertex_coord, auto &ref_map, CtInt<nb_dims> nd, const auto &children_indices, Vec<PI,d> lefs, PI i0 ) {
    Vec<PI,d+1> refs = lefs.with_pushed_value( i0 );
    std::sort( refs.begin(), refs.end() );

    // item refs => indices
    auto &rm = ref_map[ lefs.size() ];

    // get index
    PI ind = rm.find( refs, /*on new item*/ [&]( PI new_ind ) {
        // if on a vertex, add the coordinates 
        if ( d == nb_dims )
            vertex_coords << vertex_coord;

        // register new parenting vectors
        rm.on_new_ref();
    } );

    //
    for( PI e = 0; e < children_indices.size(); ++e )
        rm.parenting[ e ][ ind ].push_back_unique( children_indices[ e ] );

    // recursion
    auto new_children_indices = children_indices.with_pushed_value( ind );
    CtRange<0,d>::for_each_item( [&]( auto i ) {
        get_rec_item_data( vertex_coords, vertex_coord, ref_map, nd, new_children_indices, lefs.without_index( i ), i0 );
    } );
}
