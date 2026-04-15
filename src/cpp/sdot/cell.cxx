#pragma once

#include "../support/ASSERT.h"
#include "Cell.h"

namespace sdot {

#define UTP template<class TF,int ct_dim,class Arch,class CellInfo,class CutInfo>
#define DTP Cell<TF,ct_dim,Arch,CellInfo,CutInfo>

UTP DTP::Cell( int actual_dim ) : curr_op_id( 0 ), item_map( actual_dim + 1 ), pf( actual_dim ), df( actual_dim ) {
    if ( ct_dim >= 0 )
        ASSERT( actual_dim == ct_dim );
}

UTP void DTP::display_vtk( VtkOutput& vo ) const {
    for ( PI n0 = 0; n0 < vertices.size(); ++n0 ) {
        for ( const auto& edge : vertices[ n0 ].edge_links ) {
            const PI n1 = edge.vertex_index;
            if ( n0 < n1 ) {
                std::array<VtkOutput::Pt,2> edge{ VtkOutput::Pt( vertices[ n0 ].pos ), VtkOutput::Pt( vertices[ n1 ].pos ) };
                vo.add_edge( edge );
            }
        }
    }
}

UTP DTP DTP::axis_aligned_hypercube( Pt p0, Pt p1, CellInfo cell_info, CutInfo cut_info ) {
    const TF length = norm_2( p0 - p1 );
    const PI dim = p0.size();

    Cell res = englobing_simplex( dim, 10 * length );
    res.info = cell_info;
    for( PI d = 0; d < dim; ++d ) {
        res.cut( res.pf.value_at( d, +1 ), + p1[ d ], 2 * d + 0, cut_info );
        res.cut( res.pf.value_at( d, -1 ), - p0[ d ], 2 * d + 1, cut_info );
    }
    return res;
}

UTP DTP DTP::simplex( int dim, std::span<Pt> points, CellInfo cell_info, CutInfo cut_info ) {
    const PI nb_vertices = points.size();

    Cell res( dim );
    res.info = cell_info;

    // vertices
    res.vertices.clear();
    res.vertices.reserve( nb_vertices );
    for( PI i = 0; i < nb_vertices; ++i ) {
        res.vertices.push_back( Vertex{
            .pos = points[ i ],
            .cut_indices = res.df.with_func( [&]( PI d ) { return d + ( d >= i ); } )
        } );
    }

    // edges
    for ( PI a = 0; a < nb_vertices; ++a )
        for ( PI b = 0; b < nb_vertices; ++b )
            if ( a != b )
                res.vertices[ a ].edge_links.push_back( EdgeLink{ .num_cut_to_remove = b - ( b > a ), .vertex_index = b } );

    // cuts
    res.cuts.clear();
    res.cuts.reserve( nb_vertices );
    for ( int n0 = 0; n0 <= dim; ++n0 ) {
        const PI n1 = ( n0 + 1 ) % ( dim + 1 ); // a "random" point that is not n0
        const Pt p0 = res.vertices[ n0 ].pos;
        const Pt p1 = res.vertices[ n1 ].pos;

        // orthogonalization of p1 - p0 wrt the facing face
        Pt dir = p1 - p0;
        _for_each_2_comb_excepted( nb_vertices, n0, [ & ]( PI a, PI b ) {
            Pt cor = res.vertices[ a ].pos - res.vertices[ b ].pos;
            dir -= dot( cor, dir ) / dot( cor, cor ) * cor;
        } );

        //
        res.cuts.push_back( Cut{
            .info = cut_info,
            .dir = dir,
            .sp = dot( dir, p1 ),
            .ext = 1
        } );
    }
    return res;
}

UTP DTP DTP::axis_aligned_simplex( int dim, TF length, CellInfo cell_info, CutInfo cut_info ) {
    const PI nb_vertices = dim + 1;
    PF pf( dim );

    std::vector<Pt> points;
    points.reserve( nb_vertices );
    points.push_back( pf.zeros() );
    for ( PI num_vertex = 1; num_vertex < nb_vertices; ++num_vertex )
        points.push_back( pf.value_at( num_vertex - 1, length ) );

    return simplex( points, cell_info, cut_info );
}

UTP DTP DTP::englobing_simplex( Pt center, TF radius, CellInfo cell_info, CutInfo cut_info ) {
    const PI dim = center.size();
    PF pf( dim );

    // For a regular simplex: circumradius R = dim * insphere_radius
    const TF R = radius * dim;

    // Compute off-diagonal values a[k] and diagonal values b[k] such that:
    //   v_0        = (b[0], 0,    ..., 0)
    //   v_k (k<n)  = (a[0], ..., a[k-1], b[k], 0, ..., 0)
    //   v_n        = (a[0], ..., a[n-1])
    // All vertices have norm R and pairwise dot product -R²/dim (regular simplex centered at origin).
    Pt a( dim ), b( dim );
    TF s = 0; // s = sum_{j<k} a[j]^2
    for ( PI k = 0; k < dim; ++k ) {
        b[ k ] = std::sqrt( R * R - s );
        a[ k ] = ( -R * R / dim - s ) / b[ k ];
        s += a[ k ] * a[ k ];
    }

    // Build vertices
    std::vector<Pt> points;
    points.reserve( dim + 1 );

    points.push_back( center + pf.value_at( 0, b[ 0 ] ) );

    for ( PI k = 1; k < dim; ++k ) {
        Pt v = pf.zeros();
        for ( PI j = 0; j < k; ++j )
            v[ j ] = a[ j ];
        v[ k ] = b[ k ];
        points.push_back( center + v );
    }

    points.push_back( center + a );

    return simplex( dim, points, cell_info, cut_info );
}

UTP void DTP::_for_each_2_comb_excepted( PI size, PI excepted, auto&& func ) {
    for ( PI a = 1; a < size; ++a ) {
        if ( a == excepted )
            continue;
        for ( PI b = 0; b < a; ++b ) {
            if ( b == excepted )
                continue;
            func( a, b );
        }
    }
}

UTP void DTP::for_each_vertex( auto &&func ) const {
    for( PI index = 0; index < vertices.size(); ++index )
        func( vertices[ index ].pos, index );
}

UTP void DTP::for_each_face( auto &&/* func */ ) const {
    TODO;
}

UTP void DTP::for_each_cut( auto &&func ) const {
    for( const auto &cut : cuts )
        func( cut.dir, cut.sp, cut.id );
}

// UTP void DTP::make_simplex( TF radius, int d ) {
//     if ( d == 0 )
//         vertices[ 0 ] = pf( radius );
//     if ( d < dim )
// }

UTP void DTP::cut( const Pt& dir_cut, TF sp_cut, PI id ) {
    // get the scalar product for each node
    const PI old_vertices_size = vertices.size();
    sps.clear();
    sps.resize( vertices.size() );
    PI nb_out = 0;
    for ( PI n0 = 0; n0 < old_vertices_size; ++n0 ) {
        TF sp = dot( dir_cut, vertices[ n0 ].pos ) - sp_cut;
        nb_out += ( sp > 0 );
        sps[ n0 ] = sp;
    }

    // no change
    if ( nb_out == 0 )
        return;

    // void
    if ( nb_out == old_vertices_size ) {
        vertices.clear();
        cuts.clear();
        return;
    }

    // preparation of a face => curr_op_id + num_new_vertex map (needed to make the new edges)
    face_map.prepare_for( cuts.size() );
    ++curr_op_id;

    // old_vertex_index => new_vertex_index
    int vertex_count = 0;
    vertex_corr.clear();

    // update vertex_corr + face_map + cut edges
    for ( PI n0 = 0; n0 < old_vertices_size; ++n0 ) {
        const TF s0 = sps[ n0 ];

        // interior or exterior ?
        if ( s0 <= 0 ) { // interior
            // register vertex
            vertex_corr.push_back( vertex_count++ );

            // cut int ext edges
            for( EdgeLink &e0 : vertices[ n0 ].edge_links ) {
                const TF s1 = sps[ e0.vertex_index ];
                if ( s1 > 0 )
                    _cut_int_ext_edge( n0, e0, s0, s1 );
            }
        } else { // exterior
            // delete vertex
            vertex_corr.push_back( -1 );
        }
    }

    // we keep the new vertices :)
    for ( PI n = old_vertices_size; n < vertices.size(); ++n )
        vertex_corr.push_back( vertex_count++ );

    // compaction of the vertices
    for ( int n = 0; n < int( vertex_corr.size() ); ++n )
        if ( vertex_corr[ n ] != n && vertex_corr[ n ] >= 0 )
            vertices[ vertex_corr[ n ] ] = std::move( vertices[ n ] );
    vertices.resize( vertex_count, Vertex{ .pos = pf.zeros() } );

    // correction of the vertex references
    for( Vertex &v : vertices )
        for ( EdgeLink& e : v.edge_links )
            e.vertex_index = vertex_corr[ e.vertex_index ];

    // find unused cuts
    cut_corr.clear();
    cut_corr.resize( cuts.size(), -1 );

    // P( cuts.size() );
    // for( Vertex &v : vertices )
    //     for( PI cut_index : v.cut_indices )
    //         P( cut_index );

    for( Vertex &v : vertices )
        for( PI cut_index : v.cut_indices )
            if  ( cut_index < cut_corr.size() )
                cut_corr[ cut_index ] = 0;

    // make the cut indices corrections
    int cut_count = 0;
    for( auto &cc : cut_corr )
        if ( cc == 0 )
            cc = cut_count++;

    // remove unused cuts
    for ( int n = 0; n < int( cut_corr.size() ); ++n )
        if ( cut_corr[ n ] != n && cut_corr[ n ] >= 0 )
            cuts[ cut_corr[ n ] ] = std::move( cuts[ n ] );
    cuts.resize( cut_count, Cut{ .dir = pf.zeros() } );

    // append the new cut
    cuts.push_back( Cut{ .dir = dir_cut, .sp = sp_cut, .id = id, .ext = 0 } );
    cut_corr.push_back( cut_count++ );

    // correction of the cut references
    for( Vertex &v : vertices )
        for ( auto &cut_index : v.cut_indices )
            cut_index = cut_corr[ cut_index ];
}

UTP void DTP::_add_measure_rec( TF &res, SimpleSquareMatrix<TF,ct_dim,Arch> &M, const auto &cut_indices, PI prev_vertex_index ) const {
    using namespace std;

    if ( cut_indices.size() == 0 ) {
        res += abs( M.determinant() );
        return;
    }

    PI c = cut_indices.size();
    for( PI ind_to_remove = 0; ind_to_remove < c; ++ind_to_remove ) {
        auto new_cut_indices = cut_indices.without_index( ind_to_remove );
        ItemCorr &ic = item_map[ new_cut_indices ];
        if ( ic.vertex_index_plus_curr_op_id < curr_op_id ) {
            ic.vertex_index_plus_curr_op_id = curr_op_id + prev_vertex_index;
            continue;
        }

        const PI next_vertex_index = ic.vertex_index_plus_curr_op_id - curr_op_id;
        if ( next_vertex_index == prev_vertex_index )
            return;

        // fill the corresponding column
        for( int d = 0; d < dim(); ++d )
            M( d, c - 1 ) = vertices[ next_vertex_index ].pos[ d ] - vertices[ prev_vertex_index ].pos[ d ];

        // recursion
        _add_measure_rec( res, M, new_cut_indices, next_vertex_index );
    }
}

UTP TF DTP::measure() const {
    //
    item_map.prepare_for( cuts.size() );
    ++curr_op_id;

    SimpleSquareMatrix<TF,ct_dim,Arch> M( dim() );
    TF res = 0;

    for( PI vertex_index = 0; vertex_index < vertices.size(); ++vertex_index )
        _add_measure_rec( res, M, vertices[ vertex_index ].cut_indices, vertex_index );

    TF coeff = 1;
    for( int d = 2; d <= dim(); ++d )
        coeff *= d;

    return res / coeff;
}

UTP void DTP::_cut_int_ext_edge( PI n0, EdgeLink &e0, TF s0, TF s1 ) {
    const auto cut_indices_edge = vertices[ n0 ].cut_indices.without_index( e0.num_cut_to_remove );
    const PI n1 = e0.vertex_index;
    const Pt p0 = vertices[ n0 ].pos;
    const Pt p1 = vertices[ n1 ].pos;

    // "cut" the edge (point to the new vertex)
    const PI new_vertex_index = vertices.size();
    e0.vertex_index = new_vertex_index;

    // add the new vertex
    vertices.push_back( Vertex{
        .pos         = p0 - s0 / ( s1 - s0 ) * ( p1 - p0 ),
        .cut_indices = cut_indices_edge.with_pushed_value( cuts.size() ),
        .edge_links  = { EdgeLink{ .num_cut_to_remove = dim() - 1, .vertex_index = n0 } }
    } );

    // for each face,
    //   if the vertex is new for this face, register it
    //   else, make a new edge with the other node
    for ( PI i = 0; i < cut_indices_edge.size(); ++i ) {
        const auto cut_indices_face = cut_indices_edge.without_index( i );
        FaceCorr &fc = face_map[ cut_indices_face ];
        if ( fc.vertex_index_plus_curr_op_id < curr_op_id ) {
            fc.vertex_index_plus_curr_op_id = curr_op_id + new_vertex_index;
            fc.cut_ind_to_remove = i;
            continue;
        }

        // -> make the new edges
        const PI n_dst = fc.vertex_index_plus_curr_op_id - curr_op_id;
        const PI n_ori = new_vertex_index;

        vertices[ n_ori ].edge_links.push_back( EdgeLink{
            .num_cut_to_remove = i,
            .vertex_index = n_dst
        } );

        vertices[ n_dst ].edge_links.push_back( EdgeLink{
            .num_cut_to_remove = fc.cut_ind_to_remove,
            .vertex_index = n_ori
        } );
    }
}

UTP void DTP::check_consistency( TF eps ) const {
    using namespace std;
    for( const Vertex &v : vertices ) {
        // base integer stuff
        ASSERT( v.cut_indices.size() == dim() );
        ASSERT( v.pos.size() == dim() );
        for( PI cut_index : v.cut_indices )
            ASSERT( cut_index < cuts.size() );
        for( const EdgeLink &e : v.edge_links ) {
            ASSERT( e.num_cut_to_remove < v.cut_indices.size() );
            ASSERT( e.vertex_index < vertices.size() );
        }

        // check pos
        SimpleSquareMatrix<TF,ct_dim,Arch> m( dim() );
        DsVec<TF,ct_dim,Arch> x = pf();
        for( PI r = 0; r < dim(); ++r ) {
            for( PI c = 0; c < dim(); ++c )
                m( r, c ) = cuts[ v.cut_indices[ r ] ].dir[ c ];
            x[ r ] = cuts[ v.cut_indices[ r ] ].sp;
        }

        auto y = m.solve( x );
        for( PI r = 0; r < dim(); ++r )
            ASSERT( abs( v.pos[ r ] - y[ r ] ) <= eps );

        // check that the other points in the edge are on the edge cut
        for( const EdgeLink &e : v.edge_links ) {
            const Pt p1 = vertices[ e.vertex_index ].pos;
            for( PI num_cut : v.cut_indices.without_index( e.num_cut_to_remove ) )
                ASSERT( abs( dot( p1, cuts[ num_cut ].dir ) - cuts[ num_cut ].sp ) <= eps );
        }

    }
}

#undef UTP
#undef DTP

} // namespace sdot

