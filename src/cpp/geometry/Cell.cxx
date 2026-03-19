#pragma once

// #include "../support/P.h"
#include "Cell.h"

namespace sdot {

#define UTP template<class TF,int _dim>
#define DTP Cell<TF,_dim>

UTP DTP::Cell( int actual_dim, TF start_radius ) : pf( actual_dim ) {
    init_with_simplex( start_radius );
}

UTP void DTP::display_vtk( VtkOutput &vo ) const {
    for( PI a = 0; a < vertices.size(); ++a ) {
        for( const auto &b : vertices[ a ].links ) {
            if ( a < b ) {
                auto edge = std::array<VtkOutput::Pt,2>{ vertices[ a ].pos, vertices[ b ].pos };
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
        for( PI b = 0; b < a; ++b ) {
            vertices[ a ].links.push_back( b );
            vertices[ b ].links.push_back( a );
        }
    }

    // cuts
    for( PI num_cut = 0; num_cut <= dim; ++num_cut ) {
        PI num_oth = ( num_cut + 1 ) % ( dim + 1 );
        PT ori = vertices[ num_cut ].pos;
        PT tgt = vertices[ num_oth ].pos;
        PT dir = tgt - ori;
        for_each_2_comb_excepted( nb_vertices, num_cut, [&]( PI a, PI b ) {
            PT cor = vertices[ a ].pos - vertices[ b ].pos;
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

#undef UTP
#undef DTP

} // namespace sdot

T_Td std::ostream &operator<<( std::ostream &os, const sdot::Cell<T,d> &p ) {
    os << "vertices:";
    for( const auto &v : p.vertices )
        os << "\n  pos: " << v.pos << " cuts: " << v.cut_indices << " links: " << v.links;
    os << "\ncuts:";
    for( const auto &v : p.cuts )
        os << "\n  dir: " << v.dir << " sp: " << v.sp << " id: " << v.id << " ext: " << v.ext;
    return os;
}
