#pragma once

#include "../support/ASSERT.h"
#include "../support/TODO.h"
#include "Cell_2D.h"

namespace sdot {

#define UTP template<class TF,class Arch,class CellInfo,class CutInfo>
#define DTP Cell<TF,2,Arch,CellInfo,CutInfo>

UTP DTP::Cell( int actual_dim ) {
    ASSERT( actual_dim == ct_dim );
    edges = {
        Edge{
            .cut_dir = { 0, -1 },
            .cut_dot = 0,
            .info = {},

            .vertex_pos = { 0, 0 },
            .vertex_ext = 1
        },
        Edge{
            .cut_dir = { 1, 1 },
            .cut_dot = 1,
            .info = {},

            .vertex_pos = { 1, 0 },
            .vertex_ext = 1
        },
        Edge{
            .cut_dir = { -1, 0 },
            .cut_dot = 0,
            .info = {},

            .vertex_pos = { 0, 1 },
            .vertex_ext = 1
        }
    };
}

UTP void DTP::display_vtk( VtkOutput &vo ) const {
    for ( PI n0 = 0; n0 < edges.size(); ++n0 ) {
        const PI n1 = ( n0 + 1 ) % edges.size();
        std::array<VtkOutput::Pt,2> edge{ VtkOutput::Pt( edges[ n0 ].vertex_pos ), VtkOutput::Pt( edges[ n1 ].vertex_pos ) };
        vo.add_edge( edge );
    }
}

UTP DTP DTP::axis_aligned_hypercube( Pt p0, Pt p1, CellInfo cell_info, CutInfo cut_info ) {
    Cell res;
    res.info = cell_info;
    res.edges = {
        Edge{
            .cut_dir = { 0, -1 },
            .cut_dot = - p0[ 1 ],
            .info = cut_info,

            .vertex_pos = { p0[ 0 ], p0[ 1 ] },
            .vertex_ext = false
        },
        Edge{
            .cut_dir = { 1, 0 },
            .cut_dot = p1[ 0 ],
            .info = cut_info,

            .vertex_pos = { p1[ 0 ], p0[ 1 ] },
            .vertex_ext = false
        },
        Edge{
            .cut_dir = { 0, 1 },
            .cut_dot = p1[ 1 ],
            .info = cut_info,

            .vertex_pos = { p1[ 0 ], p1[ 1 ] },
            .vertex_ext = false
        },
        Edge{
            .cut_dir = { -1, 0 },
            .cut_dot = - p0[ 0 ],
            .info = cut_info,

            .vertex_pos = { p0[ 0 ], p1[ 1 ] },
            .vertex_ext = false
        }
    };
    return res;
}

UTP DTP DTP::simplex( int dim, std::span<Pt> points, CellInfo cell_info, CutInfo cut_info ) {
    // const PI nb_vertices = points.size();
    Cell res( dim );
    ASSERT( 0 );

    // // vertices
    // res.vertices.clear();
    // res.vertices.reserve( nb_vertices );
    // for( PI i = 0; i < nb_vertices; ++i ) {
    //     res.vertices.push_back( Vertex{
    //         .pos = points[ i ],
    //         .cut_indices = res.df.with_func( [&]( PI d ) { return d + ( d >= i ); } )
    //     } );
    // }

    // // edges
    // for ( PI a = 0; a < nb_vertices; ++a )
    //     for ( PI b = 0; b < nb_vertices; ++b )
    //         if ( a != b )
    //             res.vertices[ a ].edge_links.push_back( EdgeLink{ .num_cut_to_remove = b - ( b > a ), .vertex_index = b } );

    // // cuts
    // res.cuts.clear();
    // res.cuts.reserve( nb_vertices );
    // for ( int n0 = 0; n0 <= dim; ++n0 ) {
    //     const PI n1 = ( n0 + 1 ) % ( dim + 1 ); // a "random" point that is not n0
    //     const Pt p0 = res.vertices[ n0 ].pos;
    //     const Pt p1 = res.vertices[ n1 ].pos;

    //     // orthogonalization of p1 - p0 wrt the facing face
    //     Pt dir = p1 - p0;
    //     _for_each_2_comb_excepted( nb_vertices, n0, [ & ]( PI a, PI b ) {
    //         Pt cor = res.vertices[ a ].pos - res.vertices[ b ].pos;
    //         dir -= dot( cor, dir ) / dot( cor, cor ) * cor;
    //     } );

    //     //
    //     res.cuts.push_back( Cut{
    //         .dir = dir,
    //         .sp = dot( dir, p1 ),
    //         .id = 0,
    //         .ext = 1
    //     } );
    // }
    return res;
}

UTP DTP DTP::axis_aligned_simplex( int dim, TF length, CellInfo cell_info, CutInfo cut_info ) {
    throw std::runtime_error( "TODO" );
    // const PI nb_vertices = dim + 1;
    // PF pf( dim );

    // std::vector<Pt> points;
    // points.reserve( nb_vertices );
    // points.push_back( pf.zeros() );
    // for ( PI num_vertex = 1; num_vertex < nb_vertices; ++num_vertex )
    //     points.push_back( pf.value_at( num_vertex - 1, length ) );

    // return simplex( points );
}

UTP DTP DTP::englobing_simplex( Pt center, TF radius, CellInfo cell_info, CutInfo cut_info ) {
    throw std::runtime_error( "TODO" );
    // const PI dim = center.size();
    // PF pf( dim );

    // // For a regular simplex: circumradius R = dim * insphere_radius
    // const TF R = radius * dim;

    // // Compute off-diagonal values a[k] and diagonal values b[k] such that:
    // //   v_0        = (b[0], 0,    ..., 0)
    // //   v_k (k<n)  = (a[0], ..., a[k-1], b[k], 0, ..., 0)
    // //   v_n        = (a[0], ..., a[n-1])
    // // All vertices have norm R and pairwise dot product -R²/dim (regular simplex centered at origin).
    // Pt a( dim ), b( dim );
    // TF s = 0; // s = sum_{j<k} a[j]^2
    // for ( PI k = 0; k < dim; ++k ) {
    //     b[ k ] = std::sqrt( R * R - s );
    //     a[ k ] = ( -R * R / dim - s ) / b[ k ];
    //     s += a[ k ] * a[ k ];
    // }

    // // Build vertices
    // std::vector<Pt> points;
    // points.reserve( dim + 1 );

    // points.push_back( center + pf.value_at( 0, b[ 0 ] ) );

    // for ( PI k = 1; k < dim; ++k ) {
    //     Pt v = pf.zeros();
    //     for ( PI j = 0; j < k; ++j )
    //         v[ j ] = a[ j ];
    //     v[ k ] = b[ k ];
    //     points.push_back( center + v );
    // }

    // points.push_back( center + a );

    // return simplex( dim, points );
}

UTP void DTP::cut( const Pt &cut_dir, TF cut_dot, CutInfo cut_info ) {
    // nothing to cut
    if ( edges.empty() )
        return;

    // get the scalar product for each node
    const PI old_vertices_size = edges.size();
    PI nb_out = 0;
    for ( PI n0 = 0; n0 < old_vertices_size; ++n0 ) {
        TF sp = dot( cut_dir, edges[ n0 ].vertex_pos ) - cut_dot;
        edges[ n0 ].vertex_dot = sp;
        nb_out += ( sp > 0 );
    }

    // no change
    if ( nb_out == 0 )
        return;

    // void
    if ( nb_out == old_vertices_size ) {
        edges.clear();
        return;
    }

    // first case : need to create a new entry in edges
    if ( nb_out == 1 ) {
        // find int -> out edge
        PI n0 = 0, n1 = 0;
        for( PI n0_test = 0; n0_test < old_vertices_size; ++n0_test ) {
            const PI n1_test = ( n0_test + 1 ) % old_vertices_size;
            if ( edges[ n0_test ].inside_the_cut() && edges[ n1_test ].outside_the_cut() ) {
                n0 = n0_test;
                n1 = n1_test;
                break;
            }
        }

        const PI n2 = ( n0 + 2 ) % old_vertices_size;
        const Pt p0 = edges[ n0 ].vertex_pos;
        const Pt p1 = edges[ n1 ].vertex_pos;
        const Pt p2 = edges[ n2 ].vertex_pos;
        const TF s0 = edges[ n0 ].vertex_dot;
        const TF s1 = edges[ n1 ].vertex_dot;
        const TF s2 = edges[ n2 ].vertex_dot;

        // modify the out vertex
        Edge &v1 = edges[ n1 ];
        v1.vertex_pos = p1 - s1 / ( s2 - s1 ) * ( p2 - p1 );
        v1.vertex_ext = 0;

        // insert an item after n0
        edges.insert( edges.begin() + n1, Edge{
            .cut_dir = cut_dir,
            .cut_dot = cut_dot,
            .info = cut_info,
            .vertex_pos = p0 - s0 / ( s1 - s0 ) * ( p1 - p0 ),
            .vertex_ext = 0
        } );

        return;
    }

    // find int -> out edge (numbers are from a pentagon)
    PI n0 = 0;
    PI n1 = 0;
    for ( PI n0_test = 0; n0_test < old_vertices_size; ++n0_test ) {
        const PI n1_test = ( n0_test + 1 ) % old_vertices_size;
        if ( edges[ n0_test ].inside_the_cut() && edges[ n1_test ].outside_the_cut() ) {
            n0 = n0_test;
            n1 = n1_test;
            break;
        }
    }

    // find out -> int edge
    PI n2 = 0;
    PI n3 = 0;
    for ( PI n2_test = 0; n2_test < old_vertices_size; ++n2_test ) {
        const PI n3_test = ( n2_test + 1 ) % old_vertices_size;
        if ( edges[ n2_test ].outside_the_cut() && edges[ n3_test ].inside_the_cut() ) {
            n2 = n2_test;
            n3 = n3_test;
            break;
        }
    }

    const Pt p0 = edges[ n0 ].vertex_pos;
    const Pt p1 = edges[ n1 ].vertex_pos;
    const Pt p2 = edges[ n2 ].vertex_pos;
    const Pt p3 = edges[ n3 ].vertex_pos;
    const TF s0 = edges[ n0 ].vertex_dot;
    const TF s1 = edges[ n1 ].vertex_dot;
    const TF s2 = edges[ n2 ].vertex_dot;
    const TF s3 = edges[ n3 ].vertex_dot;

    //
    Edge &v1 = edges[ n1 ];
    v1.cut_dir = cut_dir;
    v1.cut_dot = cut_dot;
    v1.info = cut_info;
    v1.vertex_pos = p0 - s0 / ( s1 - s0 ) * ( p1 - p0 );
    v1.vertex_ext = 0;

    Edge &v2 = edges[ n2 ];
    v2.vertex_pos = p2 - s2 / ( s3 - s2 ) * ( p3 - p2 );
    v2.vertex_ext = 0;

    if ( const PI diff = nb_out - 2 ) {
        if ( n2 > n1 ) {
            edges.erase( edges.begin() + n1 + 1, edges.begin() + n2 );
        } else {
            edges.erase( edges.begin(), edges.begin() + n2 );
            edges.resize( old_vertices_size - diff );
        }
    }
}

UTP void DTP::for_each_simplex( auto &&func ) const {
    if ( edges.empty() )
        return;

    std::array<Pt,3> vertices;
    vertices[ 0 ] = edges[ 0 ].vertex_pos;
    for( PI i = 3; i <= edges.size(); ++i ) {
        vertices[ 1 ] = edges[ i - 2 ].vertex_pos;
        vertices[ 2 ] = edges[ i - 1 ].vertex_pos;
        func( vertices );
    }
}

UTP void DTP::for_each_vertex( auto &&func ) const {
    for( PI index = 0; index < edges.size(); ++index )
        func( edges[ index ].vertex_pos, index );
}

UTP void DTP::for_each_facet( auto &&func ) const {
    for( PI i = 0; i < edges.size(); ++i ) {
        const PI j = ( i + 1 ) % edges.size();
        func( Facet{ { &edges[ i ], &edges[ j ] }, edges[ i ].info } );
    }
}

UTP void DTP::for_each_face( auto &&func ) const {
    std::vector<PI> res( edges.size() );
    for( PI i = 0; i < res.size(); ++i )
        res[ i ] = i;
    func( std::move( res ) );
}

UTP void DTP::for_each_cut( auto &&func ) const {
    for( const Edge &v : edges )
        func( v.cut_dir, v.cut_dot, v.info );
}

UTP TF DTP::for_each_cut_with_measure( auto &&f ) const {
    const PI n = edges.size();
    TF sum = 0;
    for ( PI i = 0; i < n; ++i ) {
        const Pt &a = edges[ i ].vertex_pos;
        const Pt &b = edges[ ( i + 1 ) % n ].vertex_pos;
        sum += a[ 0 ] * b[ 1 ] - b[ 0 ] * a[ 1 ];

        f( edges[ i ], norm_2( b - a ) );
    }
    return sum / 2;
}

UTP auto DTP::measure( auto &&pos ) const {
    using TR = DECAYED_TYPE_OF( pos( std::array<PI,3>{ 0, 0, 0 } )[ 0 ] );
    const PI n = edges.size();
    TR sum = 0;
    for ( PI i = 0; i < n; ++i ) {
        const PI n0 = edges[ ( i + n - 1 ) % n ].info.local_dirac_index;
        const PI n1 = edges[ ( i         ) % n ].info.local_dirac_index;
        const PI n2 = edges[ ( i + 1     ) % n ].info.local_dirac_index;

        auto a = pos( std::array<PI,3>{ info.local_dirac_index, n0, n1 } );
        auto b = pos( std::array<PI,3>{ info.local_dirac_index, n1, n2 } );
        sum += a[ 0 ] * b[ 1 ] - b[ 0 ] * a[ 1 ];
    }
    return sum / 2;
}

UTP TF DTP::measure() const {
    const PI n = edges.size();
    TF sum = 0;
    for ( PI i = 0; i < n; ++i ) {
        const Pt &a = edges[ i ].vertex_pos;
        const Pt &b = edges[ ( i + 1 ) % n ].vertex_pos;
        sum += a[ 0 ] * b[ 1 ] - b[ 0 ] * a[ 1 ];
    }
    return sum / 2;
}

UTP typename DTP::Pt DTP::min_pos( Pt base_pt ) const {
    for( const Edge &edge : edges )
        base_pt = min( base_pt, edge.vertex_pos );
    return base_pt;
}

UTP typename DTP::Pt DTP::max_pos( Pt base_pt ) const {
    for( const Edge &edge : edges )
        base_pt = max( base_pt, edge.vertex_pos );
    return base_pt;
}

UTP void DTP::check_consistency( TF ) const {
}

#undef UTP
#undef DTP

} // namespace sdot

