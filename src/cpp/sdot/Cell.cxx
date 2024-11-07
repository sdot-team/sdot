#pragma once

#include <tl/support/operators/for_each_selection.h>
#include <tl/support/operators/determinant.h>
#include <tl/support/operators/norm_2.h>
#include <tl/support/operators/sp.h>

#include <tl/support/containers/SmallVec.h>
#include <tl/support/containers/CtRange.h>

#include <Eigen/Dense>
#include <limits>

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
 
    _sps.aligned_reserve( 128, CtInt<_VertexCoords::simd_size * sizeof( TF )>() );
 
    _may_have_unused_cuts = false;
    _bounded = false;
    _empty = false;

    _ref_map.for_each_item( []( auto &obj ) { obj.map.prepare_for( 128 ); } );
    _coid_ref_map = 0;

    // 0d cell has a vertex (refs should be uninitialized but valgrind may complain)
    _vertex_coords.resize( 1 );
    _vertex_refs.resize( 1 );
}

DTP UTP::Cell( const Cell &that ) : Cell( CellInfo( that.info ) ) {
    get_geometrical_data_from( that );
}

DTP void UTP::get_geometrical_data_from( const Cell &cell ) {
    _true_dimensionality = cell._true_dimensionality;

    _vertex_coords = cell._vertex_coords;
    _vertex_refs = cell._vertex_refs;
    _cuts = cell._cuts;
    _sps = cell._sps;
 
    _may_have_unused_cuts = cell._may_have_unused_cuts;
    _bounded = cell._bounded;
    _empty = cell._empty;
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

DTP Vec<Vec<TF,nb_dims>> UTP::base() const {
    if ( _true_dimensionality < nb_dims )
        return _base_vecs.slice( 0, _true_dimensionality );
    Vec<Pt> res;
    for( PI d = 0; d < nb_dims; ++d )
         res << Pt( FromFunctionOnIndex(), [&]( int i ) { return d == i; } );
    return res;
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
    for( PI num_cut = 0; num_cut < _cuts.size(); ++num_cut ) {
        if ( edge_refs.contains( num_cut ) )
            continue;
        if ( sp( _cuts[ num_cut ].dir_td( td ), tst ) > _cuts[ num_cut ].off )
            return - res;
    }
    return res;
}

DTP void UTP::remove_inactive_cuts() {
    with_ct_dim( [&]( auto td ) {
        // update active_cuts
        Vec<PI32> active_cuts( FromSizeAndItemValue(), _cuts.size(), false );
        for( auto &vertex_ref : _vertex_refs )
            for( PI d = 0; d < td; ++d )
                active_cuts[ vertex_ref[ d ] ] = true;

        // update cuts + transform active_cuts as cut index correction
        PI nb_cuts = 0;
        for( PI num_cut = 0; num_cut < _cuts.size(); ++num_cut ) {
            if ( active_cuts[ num_cut ] ) {
                if ( nb_cuts != num_cut )
                    _cuts[ nb_cuts ] = std::move( _cuts[ num_cut ] );
                active_cuts[ num_cut ] = nb_cuts++;
            }
        }

        _cuts.resize( nb_cuts );

        // num cuts
        for( auto &vertex_ref : _vertex_refs )
            for( PI d = 0; d < td; ++d )
                vertex_ref[ d ] = active_cuts[ vertex_ref[ d ] ];
    } );
} 

DTP T_i bool UTP::_has_ext_vertex( const Vec<TF,i> &dir, TF off ) {
    constexpr PI simd_size = _VertexCoords::simd_size;
    using SimdVec = _VertexCoords::SimdVec;
    CtInt<i> td;

    if ( _sps.size() < nb_vertices_true_dim() )
        _sps.aligned_resize_woc( nb_vertices_true_dim(), CtInt<_VertexCoords::simd_size * sizeof( TF )>() );

    const PI floor_of_nb_vertices = nb_vertices_true_dim() / simd_size * simd_size;
    for( PI num_vertex = 0; num_vertex < floor_of_nb_vertices; num_vertex += simd_size ) {
        TF *ptr = _vertex_coords.data() + _vertex_coords.offset( num_vertex );
        SimdVec s = SimdVec::load_aligned( ptr ) * dir[ 0 ];
        for( int d = 1; d < i; ++d )
            s += SimdVec::load_aligned( ptr + d * simd_size ) * dir[ d ];
        if ( any( s > off ) )
            return true;
    }

    for( PI num_vertex = floor_of_nb_vertices; num_vertex < nb_vertices_true_dim(); ++num_vertex )
        if ( sp( _vertex_coords.nd_at( num_vertex, td ), dir ) > off )
            return true;

    return false;
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

DTP PI UTP::_new_coid_ref_map( PI size ) const {
    // reservation for the test
    ++_coid_ref_map;

    // + room for some PI data (size)
    return std::exchange( _coid_ref_map, _coid_ref_map + size );
}

DTP void UTP::for_each_ray_and_edge( auto &&ray_func, auto &&edge_func, auto td ) const {
    if ( td != _true_dimensionality )
        return;

    const PI op_id = _new_coid_ref_map( nb_vertices_true_dim() );
    auto &edge_map = _ref_map[ CtInt<td-1>() ].map;
    edge_map.prepare_for( _cuts.size() );

    // find edges with 2 vertices
    const PI nv = nb_vertices_true_dim();
    for( PI32 n0 = 0; n0 < nv; ++n0 ) {
        CtRange<0,td>::for_each_item( [&]( auto ind_cut ) {
            auto edge_cuts = _vertex_refs[ n0 ].slice( CtInt<0>(), td ).without_index( ind_cut );
            PI &edge_op_id = edge_map[ edge_cuts ];

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
    for( PI32 n0 = 0; n0 < nv; ++n0 ) {
        CtRange<0,td>::for_each_item( [&]( auto ind_cut ) {
            auto edge_cuts = _vertex_refs[ n0 ].slice( CtInt<0>(), td ).without_index( ind_cut );
            PI &edge_op_id = edge_map[ edge_cuts ];

            if ( edge_op_id >= op_id )
                ray_func( edge_cuts, n0 );
        } );
    }
}

DTP void UTP::for_each_ray_and_edge( auto &&ray_func, auto &&edge_func ) const {
    for_each_ray_and_edge( FORWARD( ray_func ), FORWARD( edge_func ), CtInt<nb_dims>() );
}

DTP void UTP::for_each_closed_face( auto &&func ) const {
    // sibling for each vertex (index in `vertices`), for each face
    struct FaceInfo { Vec<SmallVec<PI,2>> siblings; PI start; };
    std::map<Vec<PI32,nb_dims-2>,FaceInfo,Less> face_map;
    for_each_ray_and_edge( []( auto&&, auto&& ) {}, [&]( const Vec<PI32,nb_dims-1> &edge_cuts, Span<PI32,2> vs ) {
        // for each connected face
        for( PI i = 0; i < nb_dims - 1; ++i ) {
            Vec<PI32,nb_dims-2> face_cuts = edge_cuts.without_index( i );
            auto &fi = face_map[ face_cuts ];

            // store connected vertices
            fi.siblings.resize( nb_vertices() );
            fi.siblings[ vs[ 0 ] ] << vs[ 1 ];
            fi.siblings[ vs[ 1 ] ] << vs[ 0 ];
            fi.start = vs[ 0 ];
        }
    } );

    // for each face
    Vec<PI32> vs;
    for( const auto &p: face_map ) {
        const Vec<PI32,nb_dims-2> &face_cuts = p.first;
        const FaceInfo &fi = p.second;

        // get the loop
        vs.clear();
        for( PI n = fi.start, j = 0; ; ++j ) {
            vs << n;

            const PI s = vs.size() > 1 && vs[ vs.size() - 2 ] == fi.siblings[ n ][ 0 ];
            n = fi.siblings[ n ][ s ];

            if ( n == fi.start )
                break;

            // TODO: optimize
            if ( vs.contains( n ) ) {
                vs.clear();
                break;
            }
        }

        // call f
        if ( vs.size() )
            func( face_cuts, vs );
    }
}

DTP void UTP::for_each_vertex_coord( auto &&func, auto td ) const {
    if ( td != _true_dimensionality )
        return;
    for( PI n = 0; n < nb_vertices_true_dim(); ++n )
        func( _vertex_coords.nd_at( n, td ) );
}

DTP void UTP::for_each_vertex_coord( auto &&func ) const {
    return for_each_vertex_coord( FORWARD( func ), CtInt<nb_dims>() );
}

DTP void UTP::for_each_vertex_ref( auto &&func, auto td ) const {
    if ( td != _true_dimensionality )
        return;
    for( PI n = 0; n < nb_vertices_true_dim(); ++n )
        func( _vertex_refs[ n ].slice( CtInt<0>(), td ) );
}

DTP void UTP::for_each_vertex_ref( auto &&func ) const {
    return for_each_vertex_ref( FORWARD( func ), CtInt<nb_dims>() );
}


DTP void UTP::_remove_ext_vertices( PI old_nb_vertices ) {
    if ( old_nb_vertices == 0 )
        return;

    // first phase: use of the new nodes to fill the gap by the ext nodes
    PI new_size = nb_vertices_true_dim(), i = 0;
    while( true ) {
        // find an ext vertex
        while ( ! ( _sps[ i ] > 0 ) ) {
            if ( ++i >= old_nb_vertices ) {
                _vertex_coords.resize( new_size );
                _vertex_refs.resize( new_size );
                return;
            }
        }

        // take a new node to fill the gap
        if ( new_size == old_nb_vertices )
            break;
        --new_size;

        _vertex_coords.set_item( i, _vertex_coords[ new_size ] );
        _vertex_refs[ i ] = _vertex_refs[ new_size ];

        //
        if ( ++i >= old_nb_vertices ) {
            _vertex_coords.resize( new_size );
            _vertex_refs.resize( new_size );
            return;
        }
    }

    // second phase: use of the old nodes
    while( true ) {
        // find an ext vertex
        while ( ! ( _sps[ i ] > 0 ) ) {
            if ( ++i >= new_size ) {
                _vertex_coords.resize( new_size );
                _vertex_refs.resize( new_size );
                return;
            }
        }

        // take an int node to fill the gap
        while ( true ) {
            if ( --new_size <= i ) {
                _vertex_coords.resize( new_size );
                _vertex_refs.resize( new_size );
                return;
            }
            
            if ( ! ( _sps[ new_size ] > 0 ) )
                break;
        }

        _vertex_coords.set_item( i, _vertex_coords[ new_size ] );
        _vertex_refs[ i ] = _vertex_refs[ new_size ];

        //
        if ( ++i >= new_size ) {
            _vertex_coords.resize( new_size );
            _vertex_refs.resize( new_size );
            return;
        }
    }
}

DTP void UTP::_update_bounded() {
    const PI op_id = _new_coid_ref_map( 1 );
    auto &edge_map = _ref_map[ CtInt<nb_dims-1>() ].map;
    edge_map.prepare_for( _cuts.size() );

    // mark edges with 2 vertices
    const PI nv = _vertex_coords.size();
    for( PI32 n0 = 0; n0 < nv; ++n0 ) {
        CtRange<0,nb_dims>::for_each_item( [&]( auto ind_cut ) {
            PI &edge_op_id = edge_map[ _vertex_refs[ n0 ].without_index( ind_cut ) ];

            // second (and last) time: mark the edge
            if ( edge_op_id >= op_id )
                edge_op_id = op_id - 1;
            else
                edge_op_id = op_id;
        } );
    }

    // at least one rays => not bounded
    for( PI32 n0 = 0; n0 < nv; ++n0 )
        if ( CtRange<0,nb_dims>::find_item( [&]( auto ind_cut ) { return edge_map[ _vertex_refs[ n0 ].without_index( ind_cut ) ] == op_id; } ) )
            return;

    //
    _bounded = true;
}

DTP void UTP::_unbounded_cut( const Pt &dir, TF off, CutInfo &&cut_info ) {
    if ( _empty )
        return;

    LI old_nb_vertices = nb_vertices_true_dim();
    LI ind_of_new_cut = _cuts.size();

    // update _true_dimensionality
    if ( _true_dimensionality < nb_dims ) {
        // try to make a new base vec
        Pt new_base_vec = dir;
        Pt dir_td;
        for( PI d = 0; d < _true_dimensionality; ++d ) {
            dir_td[ d ] = sp( _base_vecs[ d ], new_base_vec );
            new_base_vec -= dir_td[ d ] * _base_vecs[ d ];
        }

        // if non null, we have a new dimension
        TF n2 = norm_2_p2( new_base_vec );
        // P( n2, 10 * nb_dims * pow( std::numeric_limits<TF>::epsilon(), 2 ) );
        if ( n2 > 100 * nb_dims * pow( std::numeric_limits<TF>::epsilon(), 2 ) ) {
            // normalize
            new_base_vec /= std::sqrt( n2 );

            // update the projected dir
            dir_td[ _true_dimensionality ] = sp( new_base_vec, dir );

            // update the base
            _base_vecs[ _true_dimensionality ] = new_base_vec;

            // update the previously defined cuts
            for( _Cut &cut : _cuts )
                cut._dir_td[ _true_dimensionality ] = sp( new_base_vec, cut.dir );

            // add the new cut (which is known to be useful)
            _cuts.push_back( dir_td, std::move( cut_info ), dir, off );

            // update vertex data
            for( PI n = 0; n < _vertex_refs.size(); ++n ) { 
                // coords
                TF noff = off;
                for( PI d = 0; d < _true_dimensionality; ++d )
                    noff -= dir_td[ d ] * _vertex_coords( n, d );
                _vertex_coords( n, _true_dimensionality ) = noff / dir_td[ _true_dimensionality ];

                // refs
                _vertex_refs[ n ][ _true_dimensionality ] = ind_of_new_cut;
            }

            // say that we have a new dim
            ++_true_dimensionality;

            // if _true_dimensionality == nb_dims, move to the vertices to the canonical base
            if ( _true_dimensionality == nb_dims )
                for( PI n = 0; n < _vertex_refs.size(); ++n )
                    _vertex_coords.set_item( n, sp( _vertex_coords[ n ], _base_vecs ) );

            //
            return;
        }
    }
 
    //
    _with_ct_dim( [&]( auto td ) {
        Vec<TF,td> dir_td;
        if ( td < nb_dims ) {
            for( PI d = 0; d < td; ++d )
                dir_td[ d ] = sp( dir, _base_vecs[ d ] );
        } else
            dir_td = dir;

        // get scalar product for each vertex
        _update_sps( dir_td, off );

        // add new vertices for each edge
        bool add_the_new_cut = false;
        for_each_ray_and_edge( [&]( const auto &refs, PI32 base_vertex ) { // ray
            auto v0 = _vertex_coords.nd_at( base_vertex, td );
            auto v1 = ray_dir( refs, base_vertex );
            auto s0 = sp( v0, dir_td );
            auto s1 = sp( v1, dir_td );
            bool e0 = s0 > off;
            bool e1 = s1 > 0;
            if ( e0 != e1 ) {
                _vertex_coords << v0 + ( off - s0 ) / s1 * v1;
                _vertex_refs << refs.with_pushed_value( ind_of_new_cut );
                add_the_new_cut = true;
            }
        }, [&]( const auto &refs, const Vec<PI32,2> &num_vertices ) { // edge
            auto v0 = _vertex_coords.nd_at( num_vertices[ 0 ], td );
            auto v1 = _vertex_coords.nd_at( num_vertices[ 1 ], td );
            auto s0 = sp( v0, dir_td );
            auto s1 = sp( v1, dir_td );
            bool e0 = s0 > off;
            bool e1 = s1 > off;
            if ( e0 != e1 ) {
                _vertex_coords << v0 - s0 / ( s1 - s0 ) * ( v1 - v0 );
                _vertex_refs << refs.with_pushed_value( ind_of_new_cut );
                add_the_new_cut = true;
            }
        }, td );

        // remove vertices with sp > 0
        _remove_ext_vertices( old_nb_vertices );

        // if everything has been wiped out, we can say that the cell is empty
        if ( _vertex_coords.empty() ) {
            _empty = true;
            _cuts.clear();
            return;
        }

        // add the new cut
        if ( add_the_new_cut ) {
            _cuts.push_back( dir_td, std::move( cut_info ), dir, off );

            // check if bounded
            if ( td == nb_dims )
                _update_bounded();
        }
    }, CtInt<1>(), CtInt<nb_dims>() );
}

DTP void UTP::_bounded_cut( const Pt &dir, TF off, CutInfo &&cut_info ) {
    // if no ext vertices => no change
    if ( ! _has_ext_vertex( dir, off ) )
        return;

    // if empty, there's no edge so we need an early return
    if ( _empty )
        return;

    // store the new cut if we have at least one ext vertex
    PI new_cut = _cuts.push_back_ind( std::move( cut_info ), dir, off );

    // prepare the edge_map (edge => index of first seen vertex in the edge)
    const PI op_id = _new_coid_ref_map( nb_vertices_true_dim() );
    auto &edge_map = _ref_map[ CtInt<nb_dims-1>() ].map;
    edge_map.prepare_for( _cuts.size() );

    // get scalar product for each vertex
    _update_sps( dir, off );

    // find the vertices for each edge
    const PI32 nv = nb_vertices_true_dim();
    for( PI32 n0 = 0; n0 < nv; ++n0 ) {
        const Pt   p0 = _vertex_coords[ n0 ];
        const TF   s0 = _sps[ n0 ];
        const bool e0 = s0 > 0;

        CtRange<0,nb_dims>::for_each_item( [&]( auto ind_cut ) {
            auto edge_cuts = _vertex_refs[ n0 ].without_index( ind_cut );
            PI &edge_op_id = edge_map[ edge_cuts ];

            // if we have the 2 vertices
            if ( edge_op_id >= op_id ) {
                const PI32 n1 = edge_op_id - op_id;
                const Pt   p1 = _vertex_coords[ n1 ];
                const TF   s1 = _sps[ n1 ];
                const bool e1 = s1 > 0;

                if ( e0 != e1 ) {
                    // append the new vertex
                    auto cn = edge_cuts.with_pushed_value( new_cut );
                    auto pn = p0 - s0 / ( s1 - s0 ) * ( p1 - p0 );

                    _vertex_coords << pn;
                    _vertex_refs << cn;
                }
            } else {
                edge_op_id = op_id + n0;
            }
        } );
    }

    // remove vertices with sp > 0
    _remove_ext_vertices( nv );

    // if everything has been wiped out, we can say that the cell is empty
    if ( _vertex_coords.empty() ) {
        _empty = true;
        _cuts.clear();
        return;
    }
}

DTP void UTP::display_vtk( VtkOutput &vo ) const {
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