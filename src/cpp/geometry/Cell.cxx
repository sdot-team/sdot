#pragma once

#include "../support/ASSERT.h"
#include "../support/P.h"
#include "Cell.h"

namespace sdot {

#define UTP template<class TF,int ct_dim>
#define DTP Cell<TF,ct_dim>

UTP DTP::Cell( int actual_dim ) : curr_op_id( 0 ), item_map( actual_dim + 1 ), pf( actual_dim ), df( actual_dim ) {
    if ( ct_dim >= 0 )
        ASSERT( actual_dim == ct_dim );
}

UTP void DTP::display_vtk( VtkOutput& vo ) const {
    for ( PI n0 = 0; n0 < vertices.size(); ++n0 ) {
        for ( const auto& edge : vertices[ n0 ].edge_links ) {
            const PI n1 = edge.vertex_index;
            if ( n0 < n1 ) {
                std::array< VtkOutput::Pt, 2 > edge{ vertices[ n0 ].pos, vertices[ n1 ].pos };
                vo.add_edge( edge );
            }
        }
    }
}

UTP void DTP::init_with_simplex( std::span<Pt> points ) {
    const PI nb_vertices = points.size();
    const PI dim = this->dim();

    // vertices
    vertices.clear();
    vertices.reserve( nb_vertices );
    for( PI i = 0; i < nb_vertices; ++i ) {
        vertices.push_back( Vertex{
            .pos = points[ i ],
            .cut_indices = df.with_func( [&]( PI d ) { return d + ( d >= i ); } )
        } );
    }

    P( points.size(), vertices.size() );

    // edges
    for ( PI a = 0; a < nb_vertices; ++a )
        for ( PI b = 0; b < nb_vertices; ++b )
            if ( a != b )
                vertices[ a ].edge_links.push_back( EdgeLink{ .num_cut_to_remove = b - ( b > a ), .vertex_index = b } );

    // cuts
    cuts.clear();
    cuts.reserve( nb_vertices );
    for ( PI n0 = 0; n0 <= dim; ++n0 ) {
        const PI n1 = ( n0 + 1 ) % ( dim + 1 ); // a "random" point that is not n0
        const Pt p0 = vertices[ n0 ].pos;
        const Pt p1 = vertices[ n1 ].pos;

        // orthogonalization of p1 - p0 wrt the facing face
        Pt dir = p1 - p0;
        _for_each_2_comb_excepted( nb_vertices, n0, [ & ]( PI a, PI b ) {
            Pt cor = vertices[ a ].pos - vertices[ b ].pos;
            dir -= sp( cor, dir ) / sp( cor, cor ) * cor;
        } );

        //
        cuts.push_back( Cut{
            .dir = dir,
            .sp = sp( dir, p1 ),
            .id = 0,
            .ext = 1
        } );
    }
}

UTP void DTP::init_with_axis_aligned_simplex( TF length ) {
    const PI nb_vertices = dim() + 1;

    std::vector<Pt> points( nb_vertices );
    points[ 0 ] = pf.zeros();
    for ( PI num_vertex = 1; num_vertex < nb_vertices; ++num_vertex )
        points[ num_vertex ] = pf.value_at( num_vertex - 1, length );

    init_with_simplex( points );
}

UTP void DTP::init_with_englobing_simplex( TF radius ) {
    const PI nb_vertices = dim() + 1;

    std::vector<Pt> points( nb_vertices );
    points.push_back( pf.value_at( 0, radius ) );
    for ( PI num_vertex = 1; num_vertex < nb_vertices; ++num_vertex ) {
        for ( PI d = 0; d < num_vertex; ++d )
            points[ d ].pos[ num_vertex - 1 ] = -radius / num_vertex;
        points.push_back( pf.value_at( num_vertex - 1, radius ) );
    }

    init_with_simplex( points );
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
        TF sp = sdot::sp( dir_cut, vertices[ n0 ].pos ) - sp_cut;
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
    for ( int n = 0; n < vertex_corr.size(); ++n )
        if ( vertex_corr[ n ] != n && vertex_corr[ n ] >= 0 )
            vertices[ vertex_corr[ n ] ] = std::move( vertices[ n ] );
    vertices.resize( vertex_count );

    // correction of the vertex references
    for( Vertex &v : vertices )
        for ( EdgeLink& e : v.edge_links )
            e.vertex_index = vertex_corr[ e.vertex_index ];

    // find unused cuts
    cut_corr.clear();
    cut_corr.resize( cuts.size(), -1 );
    for( Vertex &v : vertices )
        for( PI cut_index : v.cut_indices )
            cut_corr[ cut_index ] = 0;

    // make the cut indices corrections
    int cut_count = 0;
    for( auto &cc : cut_corr )
        if ( cc == 0 )
            cc = cut_count++;

    // remove unused cuts
    for ( int n = 0; n < cut_corr.size(); ++n )
        if ( cut_corr[ n ] != n && cut_corr[ n ] >= 0 )
            cuts[ cut_corr[ n ] ] = std::move( cuts[ n ] );
    cuts.resize( cut_count );

    // append the new cut
    cuts.push_back( Cut{ .dir = dir_cut, .sp = sp_cut, .id = id, .ext = 0 } );
    cut_corr.push_back( cut_count++ );

    // correction of the cut references
    for( Vertex &v : vertices )
        for ( auto &cut_index : v.cut_indices )
            cut_index = cut_corr[ cut_index ];
}

UTP void DTP::_add_measure_rec( TF &res, SimpleSquareMatrix<TF,ct_dim> &M, const auto &cut_indices, PI prev_vertex_index ) const {
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

    SimpleSquareMatrix<TF,ct_dim> M( dim() );
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
        .edge_links  = { EdgeLink{ .vertex_index = n0, .num_cut_to_remove = dim() - 1 } }
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
        SimpleSquareMatrix<TF,ct_dim> m( dim() );
        Point<TF,ct_dim> x = pf();
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
                ASSERT( abs( sp( p1, cuts[ num_cut ].dir ) - cuts[ num_cut ].sp ) <= eps );
        }

    }
}

#undef UTP
#undef DTP

} // namespace sdot

T_Td std::ostream& operator<<( std::ostream& os, const sdot::Cell< T, d >& p ) {
    os << "vertices:";
    for ( const auto& v : p.vertices ) {
        os << "\n  pos: " << v.pos << " cuts: " << v.cut_indices << " edge: ";
        for ( const auto& edge : v.edge_links ) {
            os << "[ n: " << edge.vertex_index << ", c: ";
            for ( sdot::PI i = 0, c = 0; i < v.cut_indices.size(); ++i )
                if ( i != edge.num_cut_to_remove )
                    os << ( c++ ? ", " : "" ) << v.cut_indices[ i ];
            os << " ]";
        }
    }
    os << "\ncuts:";
    for ( const auto& v : p.cuts )
        os << "\n  dir: " << v.dir << " sp: " << v.sp << " id: " << v.id << " ext: " << v.ext;
    return os;
}
