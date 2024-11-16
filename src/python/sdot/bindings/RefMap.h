#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/compare.h>
#include <unordered_map>

struct DiracVecHash { 
    PI operator()( const auto &vr ) const {
        PI m = 1, r = 0;
        for( auto v : vr ) {
            r += m * v;
            m *= nb_diracs;
        }
        return r;
    }
    PI nb_diracs;
};

template<int dim> struct RefMapForDim {
    RefMapForDim( PI nb_diracs ) : map( nb_diracs, DiracVecHash( nb_diracs ) ) {
    }

    PI index( const Vec<PI,dim> vr, auto &&on_new_item ) {
        auto iter = map.find( vr );
        if ( iter == map.end() ) {
            iter = map.insert( iter, { vr, refs.size() } );
            on_new_item();
            refs << vr;
        }
        return iter->second;
    }

    std::unordered_map<Vec<PI,dim>,PI,DiracVecHash,Equal> map;
    Vec<Vec<PI,dim>> refs;
};

template<> struct RefMapForDim<1> {
    RefMapForDim( PI nb_diracs ) {
    }

    PI index( const Vec<PI,1> vr, auto &&on_new_item ) {
        return vr[ 1 ];
    }
};

template<class T,int d> void for_each_subvec( const Vec<T,d> &refs, auto &&func ) {
    if constexpr ( d ) {
        func( refs );
        for( PI i = 0; i < d; ++i )
            for_each_subvec( refs.without_index( i ), func );
    }
}
