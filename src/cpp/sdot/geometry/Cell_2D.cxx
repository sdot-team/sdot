#pragma once

#include "../support/ASSERT.h"
#include "../support/P.h"
#include "Cell_2D.h"

namespace sdot {

#define UTP template<class TF,class Arch>
#define DTP Cell<TF,2,Arch>

UTP DTP::Cell( int actual_dim ) {
    ASSERT( actual_dim == ct_dim );
    vertices = {
        VertexAndCut{
            .cut_dir = { 0, -1 },
            .cut_dot = 0,
            .cut_id = 0,
            .pos = { 0, 0 },
            .ext = 1
        },
        VertexAndCut{
            .cut_dir = { 1, 1 },
            .cut_dot = 1,
            .cut_id = 0,
            .pos = { 1, 0 },
            .ext = 1
        },
        VertexAndCut{
            .cut_dir = { -1, 0 },
            .cut_dot = 0,
            .cut_id = 0,
            .pos = { 0, 1 },
            .ext = 1
        }
    };
}

UTP void DTP::display_vtk( VtkOutput &vo ) const {
    for ( PI n0 = 0; n0 < vertices.size(); ++n0 ) {
        const PI n1 = ( n0 + 1 ) % vertices.size();
        std::array<VtkOutput::Pt,2> edge{ VtkOutput::Pt( vertices[ n0 ].pos ), VtkOutput::Pt( vertices[ n1 ].pos ) };
        vo.add_edge( edge );
    }
}

UTP DTP DTP::axis_aligned_hypercube( Pt p0, Pt p1, bool ext ) {
    Cell res;
    res.vertices = {
        VertexAndCut{
            .cut_dir = { 0, -1 },
            .cut_dot = - p0[ 1 ],
            .cut_id = 0,
            .pos = { p0[ 0 ], p0[ 1 ] },
            .ext = ext
        },
        VertexAndCut{
            .cut_dir = { 1, 0 },
            .cut_dot = p1[ 0 ],
            .cut_id = 0,
            .pos = { p1[ 0 ], p0[ 1 ] },
            .ext = ext
        },
        VertexAndCut{
            .cut_dir = { 0, 1 },
            .cut_dot = p1[ 1 ],
            .cut_id = 0,
            .pos = { p1[ 0 ], p1[ 1 ] },
            .ext = ext
        },
        VertexAndCut{
            .cut_dir = { -1, 0 },
            .cut_dot = - p0[ 0 ],
            .cut_id = 0,
            .pos = { p0[ 0 ], p1[ 1 ] },
            .ext = ext
        }
    };
    return res;
}

UTP DTP DTP::simplex( int dim, std::span<Pt> points ) {
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

UTP DTP DTP::axis_aligned_simplex( int dim, TF length ) {
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

UTP DTP DTP::englobing_simplex( Pt center, TF radius ) {
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

UTP void DTP::cut( const Pt &cut_dir, TF cut_dot, PI cut_id ) {
    // nothing to cut
    if ( vertices.empty() )
        return;

    // get the scalar product for each node
    const PI old_vertices_size = vertices.size();
    PI nb_out = 0;
    for ( PI n0 = 0; n0 < old_vertices_size; ++n0 ) {
        TF sp = sdot::dot( cut_dir, vertices[ n0 ].pos ) - cut_dot;
        vertices[ n0 ].sp = sp;
        nb_out += ( sp > 0 );
    }

    // no change
    if ( nb_out == 0 )
        return;

    // void
    if ( nb_out == old_vertices_size ) {
        vertices.clear();
        return;
    }

    // first case : need to create a new entry in vertices
    if ( nb_out == 1 ) {
        // find int -> out edge
        PI n0 = 0, n1 = 0;
        for( PI n0_test = 0; n0_test < old_vertices_size; ++n0_test ) {
            const PI n1_test = ( n0_test + 1 ) % old_vertices_size;
            if ( vertices[ n0_test ].inside_the_cut() && vertices[ n1_test ].outside_the_cut() ) {
                n0 = n0_test;
                n1 = n1_test;
                break;
            }
        }

        const PI n2 = ( n0 + 2 ) % old_vertices_size;
        const Pt p0 = vertices[ n0 ].pos;
        const Pt p1 = vertices[ n1 ].pos;
        const Pt p2 = vertices[ n2 ].pos;
        const TF s0 = vertices[ n0 ].sp;
        const TF s1 = vertices[ n1 ].sp;
        const TF s2 = vertices[ n2 ].sp;

        // modify the out vertex
        VertexAndCut &v1 = vertices[ n1 ];
        v1.pos = p1 - s1 / ( s2 - s1 ) * ( p2 - p1 );
        v1.ext = 0;

        // insert an item after n0
        vertices.insert( vertices.begin() + n1, VertexAndCut{
            .cut_dir = cut_dir,
            .cut_dot = cut_dot,
            .cut_id = cut_id,
            .pos = p0 - s0 / ( s1 - s0 ) * ( p1 - p0 ),
            .ext = 0
        } );

        return;
    }

    // find int -> out edge (numbers are from a pentagon)
    PI n0 = 0;
    PI n1 = 0;
    for ( PI n0_test = 0; n0_test < old_vertices_size; ++n0_test ) {
        const PI n1_test = ( n0_test + 1 ) % old_vertices_size;
        if ( vertices[ n0_test ].inside_the_cut() && vertices[ n1_test ].outside_the_cut() ) {
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
        if ( vertices[ n2_test ].outside_the_cut() && vertices[ n3_test ].inside_the_cut() ) {
            n2 = n2_test;
            n3 = n3_test;
            break;
        }
    }

    const Pt p0 = vertices[ n0 ].pos;
    const Pt p1 = vertices[ n1 ].pos;
    const Pt p2 = vertices[ n2 ].pos;
    const Pt p3 = vertices[ n3 ].pos;
    const TF s0 = vertices[ n0 ].sp;
    const TF s1 = vertices[ n1 ].sp;
    const TF s2 = vertices[ n2 ].sp;
    const TF s3 = vertices[ n3 ].sp;

    //
    VertexAndCut &v1 = vertices[ n1 ];
    v1.cut_dir = cut_dir;
    v1.cut_dot = cut_dot;
    v1.cut_id = cut_id;
    v1.pos = p0 - s0 / ( s1 - s0 ) * ( p1 - p0 );
    v1.ext = 0;

    VertexAndCut &v2 = vertices[ n2 ];
    v2.pos = p2 - s2 / ( s3 - s2 ) * ( p3 - p2 );
    v2.ext = 0;

    if ( const PI diff = nb_out - 2 ) {
        if ( n2 > n1 ) {
            vertices.erase( vertices.begin() + n1 + 1, vertices.begin() + n2 );
        } else {
            vertices.erase( vertices.begin(), vertices.begin() + n2 );
            vertices.resize( old_vertices_size - diff );
        }
    }
}

UTP void DTP::for_each_cut( auto &&func ) {
    for( VertexAndCut &v : vertices )
        func( v.cut_dir, v.cut_dot, v.cut_id );
}

UTP TF DTP::measure() const {
    ASSERT( 0 );
}

UTP void DTP::check_consistency( TF eps ) const {
}

#undef UTP
#undef DTP

} // namespace sdot

template<class TF,class Arch>
std::ostream& operator<<( std::ostream &os, const sdot::Cell<TF,2,Arch> &p ) {
    for ( const auto &v : p.vertices )
        os << "\n  pos: " << v.pos << " dir: " << v.cut_dir << " dot: " << v.cut_dot;
    return os;
}
