#pragma once

#include <tl/support/operators/for_each_selection.h>
#include <tl/support/operators/determinant.h>
#include <tl/support/operators/norm_2.h>
#include <tl/support/operators/sp.h>

#include <tl/support/containers/CtRange.h>

#include <Eigen/Dense>

#include "Cell.h"

#include <tl/support/P.h>

namespace sdot {

#define DTP template<class Arch,class TF,int nb_dims,class CutInfo,class CellInfo>
#define UTP Cell<Arch,TF,nb_dims,CutInfo,CellInfo>

DTP UTP::Cell( CellInfo &&info ) : info( std::move( info ) ) {
    _true_dimensionality = 0;

    _vertex_coords.reserve( 128 );
    _vertex_refs.reserve( 128 );
    _cuts.reserve( 128 );
    _sps.reserve( 128 );
 
    _may_have_unused_cuts = false;
    _bounded = false;
    _empty = false;

    _ref_map.for_each_item( []( auto &obj ) { obj.map.prepare_for( 128 ); } );

    // 0d cell has a vertex (refs should be uninitialized but valgrind may complain)
    _vertex_coords.resize( 1 );
    _vertex_refs.resize( 1 );
}

DTP T_i auto UTP::_with_ct_dim( auto &&func, CtInt<i> min_td, CtInt<i> max_td ) const {
    return func( max_td );
}

DTP T_ij auto UTP::_with_ct_dim( auto &&func, CtInt<i> min_td, CtInt<j> max_td ) const {
    if ( j == _true_dimensionality )
        return func( max_td );
    else
        return _with_ct_dim( FORWARD( func ), min_td, CtInt<j-1>() );
}

DTP T_i auto UTP::_with_ct_dim( auto &&func, CtInt<i> min_td, CtInt<i> max_td ) {
    return func( max_td );
}

DTP T_ij auto UTP::_with_ct_dim( auto &&func, CtInt<i> min_td, CtInt<j> max_td ) {
    if ( j == _true_dimensionality )
        return func( max_td );
    else
        return _with_ct_dim( FORWARD( func ), min_td, CtInt<j-1>() );
}

DTP auto UTP::with_ct_dim( auto &&func ) const {
    return _with_ct_dim( FORWARD( func ), CtInt<0>(), CtInt<nb_dims>() );
}

DTP auto UTP::with_ct_dim( auto &&func ) {
    return _with_ct_dim( FORWARD( func ), CtInt<0>(), CtInt<nb_dims>() );
}

DTP T_i Vec<PI32,i> UTP::vertex_refs( PI num_vertex, CtInt<i> choosen_nb_dims ) const {
    return _vertex_refs[ num_vertex ];
}

DTP Vec<PI32,nb_dims> UTP::vertex_refs( PI num_vertex ) const {
    return _vertex_refs[ num_vertex ];
}

DTP T_i Vec<TF,i> UTP::vertex_coord( PI num_vertex, CtInt<i> choosen_nb_dims ) const {
    return _vertex_coords.nd_at( num_vertex, choosen_nb_dims );
}

DTP UTP::Pt UTP::vertex_coord( PI num_vertex ) const {
    return _vertex_coords[ num_vertex ];
}

DTP PI UTP::nb_stored_cuts() const {
    return _cuts.size();
}

DTP PI UTP::nb_active_cuts() const {
    if ( _may_have_unused_cuts ) {
        Vec<bool> cut_is_active( FromSizeAndItemValue(), _cuts.size(), false );
        with_ct_dim( [&]( auto td ) {
            for( PI n = 0; n < nb_vertices_true_dim(); ++n )
                for( PI32 num : vertex_refs( n, td ) )
                    cut_is_active[ num ] = true;
        } );

        PI res = 0;
        for( PI n = 0; n < _cuts.size(); ++n )
            res += cut_is_active[ n ];
        return res;
    }

    return nb_stored_cuts();
}

DTP PI UTP::nb_vertices_true_dim() const {
    return _vertex_coords.size();
}

DTP PI UTP::nb_vertices() const {
    return _true_dimensionality == nb_dims ? _vertex_coords.size() : 0;
}

DTP T_i Vec<TF,i+1> UTP::ray_dir( const Vec<LI,i> &edge_refs, LI base_vertex ) const {
    Vec<Vec<TF,i>,i> M;
    Vec<TF,i+1> res;
    CtInt<i+1> td;
    for( LI d = 0; d <= i; ++d ) {
        for( LI c = 0; c < i; ++c )
            for( LI r = 0; r < i; ++r )
                M[ r ][ c ] = _cuts[ edge_refs[ c ] ].dir_td( td, r + ( r >= d ) );
        res[ d ] = d & 1 ? - determinant( M ) : determinant( M );
    }

    Vec<TF,i+1> tst = _vertex_coords.nd_at( base_vertex, td ) + res;
    for( const _Cut &cut : _cuts )
        if ( sp( cut.dir_td( td ), tst ) > cut.off_td( td ) )
            return - res;
    return res;
}

DTP T_i void UTP::_update_sps( const Vec<TF,i> &dir, TF off ) {
    constexpr PI simd_size = _VertexCoords::simd_size;
    using SimdVec = _VertexCoords::SimdVec;
    CtInt<i> td;

    if ( _sps.size() < nb_vertices_true_dim() )
        _sps.aligned_resize_woc( nb_vertices_true_dim(), CtInt<_VertexCoords::simd_size * sizeof( TF )>() );

    const PI floor_of_nb_vertices = nb_vertices_true_dim() / simd_size * simd_size;
    for( PI num_vertex = 0; num_vertex < floor_of_nb_vertices; num_vertex += simd_size ) {
        TF *ptr = _vertex_coords.data() + _vertex_coords.offset( num_vertex );
        SimdVec s = SimdVec::load_aligned( ptr ) * dir[ 0 ] - off;
        for( int d = 1; d < i; ++d )
            s += SimdVec::load_aligned( ptr + d * simd_size ) * dir[ d ];
        s.store_aligned( _sps.data() + num_vertex );
    }

    for( PI num_vertex = floor_of_nb_vertices; num_vertex < nb_vertices_true_dim(); ++num_vertex )
        _sps[ num_vertex ] = sp( _vertex_coords.nd_at( num_vertex, td ), dir ) - off;
}

DTP PI UTP::_new_coid_ref_map( PI size ) {
    // reservation for the test
    ++_coid_ref_map;

    // + room for some PI data (size)
    return std::exchange( _coid_ref_map, _coid_ref_map + size );
}

DTP void UTP::_for_each_ray_and_edge( auto &&ray_func, auto &&edge_func, auto td ) {
    const PI op_id = _new_coid_ref_map( nb_vertices_true_dim() );
    auto &edge_map = _ref_map[ CtInt<td-1>() ].map;
    edge_map.prepare_for( _cuts.size() );
    P( _cuts.size() );

    // find edges with 2 vertices
    for( PI32 n0 = 0; n0 < nb_vertices_true_dim(); ++n0 ) {
        CtRange<0,td>::for_each_item( [&]( auto ind_cut ) {
            auto edge_cuts = _vertex_refs[ n0 ].slice( CtInt<0>(), td ).without_index( ind_cut );
            PI &edge_op_id = edge_map[ edge_cuts ];
            P( td, edge_cuts );

            if ( edge_op_id >= op_id ) {
                Vec<PI32,2> ns{ PI32( edge_op_id - op_id ), n0 };
                edge_func( edge_cuts, ns );

                // mark the edge (to say that it's not a ray)
                edge_op_id = op_id - 1;
            } else {
                edge_op_id = op_id + n0;
            }
        } );
    }

    // find rays
    for( PI32 n0 = 0; n0 < nb_vertices_true_dim(); ++n0 ) {
        CtRange<0,td>::for_each_item( [&]( auto ind_cut ) {
            auto edge_cuts = _vertex_refs[ n0 ].slice( CtInt<0>(), td ).without_index( ind_cut );
            PI &edge_op_id = edge_map[ edge_cuts ];

            if ( edge_op_id >= op_id )
                ray_func( edge_cuts, n0 );
        } );
    }
}

DTP void UTP::_unbounded_cut( const Pt &dir, TF off, CutInfo &&cut_info ) {
    // update _true_dimensionality
    if ( _true_dimensionality < nb_dims ) {
        // try to make a new base vec
        Pt new_base_vec = dir;
        Pt proj_dir;
        for( PI d = 0; d < _true_dimensionality; ++d ) {
            proj_dir[ d ] = sp( _base_vecs[ d ], new_base_vec ) / norm_2_p2( _base_vecs[ d ] );
            new_base_vec -= proj_dir[ d ] * _base_vecs[ d ];
        }

        // if non null, we have a new dimension
        if ( TF n2 = norm_2_p2( new_base_vec ) ) {
            new_base_vec /= std::sqrt( n2 );

            proj_dir[ _true_dimensionality ] = sp( new_base_vec, dir );

            // add the cut (which is known to be useful)
            PI new_ref = _cuts.push_back_ind( std::move( cut_info ), dir, off );

            // update vertex data
            for( PI n = 0; n < _vertex_refs.size(); ++n ) { 
                // coords
                TF noff = off;
                for( PI d = 0; d < _true_dimensionality; ++d )
                    noff -= proj_dir[ d ] * _vertex_coords( n, d );
                _vertex_coords( n, _true_dimensionality ) = noff / proj_dir[ _true_dimensionality ];

                // refs
                _vertex_refs[ n ][ _true_dimensionality ] = new_ref;
            }

            // update the base
            _base_vecs[ _true_dimensionality ] = new_base_vec;

            //
            ++_true_dimensionality;

            // if _true_dimensionality == nb_dims, move to the canonical base
            if ( _true_dimensionality == nb_dims )
                for( PI n = 0; n < _vertex_refs.size(); ++n )
                    _vertex_coords.set_item( n, sp( _vertex_coords[ n ], _base_vecs ) );

            //
            return;
        }

        //
    }

    //
    LI old_nb_vertices = nb_vertices_true_dim();
 
    //
    _with_ct_dim( [&]( auto td ) {
        Vec<TF,td> dir_td = dir;
        TF off_td = off;
        if ( td < nb_dims )
            TODO;

        // get scalar product for each vertex
        _update_sps( dir_td, off_td );

        // add new vertices for each edge
        _for_each_ray_and_edge( [&]( const auto &refs, PI32 base_vertex ) { // ray
            auto vp = _vertex_coords.nd_at( base_vertex, td );
            auto rd = ray_dir( refs, base_vertex );
            TF alpha = sp( rd, dir_td );
            if ( alpha > 0 ) {
                Vec<TF,td> pos = vp + ( off - sp( vp, dir_td ) ) / alpha * rd;
                _vertex_coords << pos;
                _vertex_refs << refs;
            }
        }, [&]( const auto &refs, const Vec<PI32,2> &num_vertices ) { // edge
            auto v0 = _vertex_coords.nd_at( num_vertices[ 0 ], td );
            auto v1 = _vertex_coords.nd_at( num_vertices[ 1 ], td );
            auto s0 = sp( v0, dir_td );
            auto s1 = sp( v1, dir_td );
            bool e0 = s0 > off_td;
            bool e1 = s1 > off_td;
            if ( e0 != e1 ) {
                Vec<TF,td> pos = v0 - s0 / ( s1 - s0 ) * ( v1 - v0 );
                _vertex_coords << pos;
                _vertex_refs << refs;
            }
        }, td );

        // add the new cut (we will check if useful later)
        _Cut *new_cut = _cuts.push_back( std::move( cut_info ), dir, off );
        if ( td < nb_dims ) {
            for( PI d = 0; d < td; ++d )
                new_cut->_dir_td[ d ] = dir_td[ d ];
            new_cut->_off_td = off_td;
        }

        // remove unused vertices
    }, CtInt<1>(), CtInt<nb_dims>() );
}

DTP void UTP::_bounded_cut( const Pt &dir, TF off, CutInfo &&cut_info ) {
    TODO;
}

DTP void UTP::display( Displayer &ds ) const {
    ds.start_object();

    ds.append_attribute( "true_dimensionality", _true_dimensionality );
    if ( _true_dimensionality < nb_dims )
        ds.append_attribute( "base", _base_vecs.slice( 0, _true_dimensionality ) );
    ds.append_attribute( "vertex_coords", map_vec( _vertex_coords, [&]( const auto &v ) -> Vec<TF> { return v.slice( 0, _true_dimensionality ); } ) );
    ds.append_attribute( "vertex_refs", map_vec( _vertex_refs, [&]( const auto &v ) -> Vec<LI> { return v.slice( 0, _true_dimensionality ); } ) );

    ds.end_object();
}

DTP void UTP::cut( const Pt &dir, TF off, CutInfo &&cut_info ) {
    return _bounded ? _bounded_cut( dir, off, std::move( cut_info ) ) : _unbounded_cut( dir, off, std::move( cut_info ) ) ;
}

#undef DTP
#undef UTP

} // namespace sdot