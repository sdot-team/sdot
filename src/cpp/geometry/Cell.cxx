#pragma once

#include "../support/ASSERT.h"
#include "../support/P.h"
#include "Cell.h"

namespace sdot {

#define UTP template<class TF,int _dim>
#define DTP Cell<TF,_dim>

UTP DTP::Cell( int actual_dim, TF start_radius ) : curr_op_id( 0 ), pf( actual_dim ) {
    init_with_simplex( start_radius );
}

UTP void DTP::display_vtk( VtkOutput &vo ) const {
    for( PI n0 = 0; n0 < vertices.size(); ++n0 ) {
        for( const auto &edge : vertices[ n0 ].edge_links ) {
            const PI n1 = edge.vertex_index;
            if ( n0 < n1 ) {
                std::array<VtkOutput::Pt,2> edge{ vertices[ n0 ].pos, vertices[ n1 ].pos };
                vo.add_edge( edge );
            }
        }
    }
}

UTP void DTP::init_with_simplex( TF start_radius ) {
    const PI nb_vertices = dim() + 1;
    const PI dim = this->dim();

    vertices.reserve( nb_vertices );
    cuts.reserve( nb_vertices );

    // vertices pos
    vertices.push_back( Vertex{ .pos = pf.value_at( 0, start_radius ), .cut_indices = { dim } } );
    for( PI num_vertex = 1; num_vertex < nb_vertices; ++num_vertex ) {
        for( PI d = 0; d < num_vertex; ++d )
            vertices[ d ].pos[ num_vertex - 1 ] = - start_radius / num_vertex;
        vertices.push_back( Vertex{ .pos = pf.value_at( num_vertex - 1, start_radius ), .cut_indices = { dim } } );
    }

    // cut indices
    for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
        for( PI d = 0; d < dim; ++d )
            vertices[ num_vertex ].cut_indices[ d ] = d + ( d >= num_vertex );

    // edges
    for( PI a = 0; a < nb_vertices; ++a ) {
        for( PI b = 0; b < nb_vertices; ++b ) {
            if ( a != b ) {
                vertices[ a ].edge_links.push_back( EdgeLink{
                    .num_cut_to_remove = b - ( b > a ),
                    .vertex_index = b
                } );
            }
        }
    }

    // cuts
    for( PI num_cut = 0; num_cut <= dim; ++num_cut ) {
        PI num_oth = ( num_cut + 1 ) % ( dim + 1 );
        Pt ori = vertices[ num_cut ].pos;
        Pt tgt = vertices[ num_oth ].pos;
        Pt dir = tgt - ori;
        for_each_2_comb_excepted( nb_vertices, num_cut, [&]( PI a, PI b ) {
            Pt cor = vertices[ a ].pos - vertices[ b ].pos;
            dir -= sp( cor, dir ) / sp( cor, cor ) * cor;
        } );
        cuts.push_back( Cut{ 
            .dir = dir,
            .sp = sp( dir, tgt ),
            .id = 0,
            .ext = 1
        } );
    }        
}

UTP void DTP::for_each_2_comb_excepted( PI size, PI excepted, auto &&func ) {
    for( PI a = 1; a < size; ++a ) {
        if ( a == excepted )
            continue;
        for( PI b = 0; b < a; ++b ) {
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

UTP void DTP::cut( const Pt &dir, TF sp, PI id ) {
    //
    ++curr_op_id;
    
    sps.resize( 0 );
    sps.resize( vertices.size() );
    
    const PI new_cut_index = cuts.size();
    bool used_cut = false;
    for( PI n0 = 0; n0 < vertices.size(); ++n0 ) {
        Vertex &v = vertices[ n0 ];
        const Pt p0 = v.pos;

        const TF s0 = sdot::sp( dir, p0 ) - sp;
        sps[ n0 ] = s0;

        // if ext vertex
        if ( v.s > 0 ) {
            // using FaceId = std::array<PI,dim>;
            // std::map<FaceId,PI> face_map; // -> prev vertex index
            // for each edge to an int vertex
            for( const EdgeLink &edge : v.edge_links ) {
                const TF s1 = sps[ edge.vertex_index ];
                if ( s1 <= 0 ) {
                    // add a new vertex
                    // const PI new_vertex_index = vertices.size();
                    const Pt p1 = vertices[ edge.vertex_index ].pos;
                    vertices.push_back( {
                        .pos = p0 - s0 / ( s1 - s0 ) * ( p1 - p0 ),
                        .cut_indices = vertices[ n0 ].cut_indices.without_index( edge.num_cut_to_remove ).with_pushed_value( new_cut_index ),
                        .links = {
                            EdgeLink{
                                .vertex_index = n0,
                                .num_cut_to_remove = edge.num_cut_to_remove
                            }
                        }
                    } );
                    used_cut = true;

                    //
                    // FaceId fi = 
                }
            }
        }
    }

    if ( used_cut ) {
        cuts.push_back( Cut{
            .dir = dir,
            .sp = sp,
            .id = id,
            .ext = 0
        } );
    }
}

#undef UTP
#undef DTP

} // namespace sdot

T_Td std::ostream &operator<<( std::ostream &os, const sdot::Cell<T,d> &p ) {
    os << "vertices:";
    for( const auto &v : p.vertices ) {
        os << "\n  pos: " << v.pos << " cuts: " << v.cut_indices << " edge: ";
        for( const auto &edge : v.edge_links ) {
            os << "[ n: " << edge.vertex_index << ", c: ";
                for( sdot::PI i = 0, c = 0; i < v.cut_indices.size(); ++i )
                    if ( i != edge.num_cut_to_remove )
                        os << ( c++ ? ", " : "" ) << v.cut_indices[ i ];
            os << " ]";
        }
    }
    os << "\ncuts:";
    for( const auto &v : p.cuts )
        os << "\n  dir: " << v.dir << " sp: " << v.sp << " id: " << v.id << " ext: " << v.ext;
    return os;
}
