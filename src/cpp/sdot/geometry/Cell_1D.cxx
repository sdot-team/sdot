#pragma once

#include "../support/ASSERT.h"
#include "../support/P.h"
#include "Cell_1D.h"

namespace sdot {

#define UTP template<class TF,class Arch,class CellInfo,class CutInfo>
#define DTP Cell<TF,1,Arch,CellInfo,CutInfo>

UTP DTP::Cell( int actual_dim ) {
    ASSERT( actual_dim == ct_dim );
    bounds = {
        Bound{
            .cut_dir = { -1 },
            .cut_dot = 1,
            .info = {},

            .vertex_pos = { -1 },
            .vertex_ext = 1
        },
        Bound{
            .cut_dir = { +1 },
            .cut_dot = 1,
            .info = {},

            .vertex_pos = { +1 },
            .vertex_ext = 1
        },
    };
}

UTP void DTP::display_vtk( VtkOutput &vo ) const {
    std::array<VtkOutput::Pt,2> edge{ VtkOutput::Pt( bounds[ 0 ].vertex_pos ), VtkOutput::Pt( bounds[ 1 ].vertex_pos ) };
    vo.add_edge( edge );
}

UTP DTP DTP::axis_aligned_hypercube( Pt p0, Pt p1, CellInfo cell_info, CutInfo cut_info ) {
    Cell res;
    res.info = cell_info;
    res.bounds = {
        Bound{
            .cut_dir = { -1 },
            .cut_dot = - p0[ 0 ],
            .info = cut_info,

            .vertex_pos = { p0[ 0 ] },
            .vertex_ext = false
        },
        Bound{
            .cut_dir = { +1 },
            .cut_dot = p1[ 0 ],
            .info = cut_info,

            .vertex_pos = { p1[ 0 ] },
            .vertex_ext = false
        },
    };
    return res;
}

UTP void DTP::cut( const Pt &cut_dir, TF cut_dot, CutInfo cut_info ) {
    TF x = cut_dot / cut_dir[ 0 ];
    if ( cut_dir[ 0 ] > 0 ) {
        if ( bounds[ 1 ].vertex_ext || bounds[ 1 ].vertex_pos[ 0 ] > x ) {
            bounds[ 1 ].cut_dir = cut_dir;
            bounds[ 1 ].cut_dot = cut_dot;
            bounds[ 1 ].info = cut_info;

            bounds[ 1 ].vertex_pos[ 0 ] = x;
            bounds[ 1 ].vertex_ext = false;
        }
    } else {
        if ( bounds[ 0 ].vertex_ext || bounds[ 0 ].vertex_pos[ 0 ] < x ) {
            bounds[ 0 ].cut_dir = cut_dir;
            bounds[ 0 ].cut_dot = cut_dot;
            bounds[ 0 ].info = cut_info;

            bounds[ 0 ].vertex_pos[ 0 ] = x;
            bounds[ 0 ].vertex_ext = false;
        }
    }
}

UTP void DTP::for_each_vertex( auto &&func ) const {
    func( bounds[ 0 ].vertex_pos, 0 );
    func( bounds[ 1 ].vertex_pos, 1 );
}

UTP void DTP::for_each_cut( auto &&func ) const {
    for( const Bound &v : bounds )
        func( v.cut_dir, v.cut_dot, v.info );
}

UTP TF DTP::for_each_cut_with_measure( auto &&f ) const {
    for( const Bound &v : bounds )
        func( v, 1 );
    return measure();
}

UTP auto DTP::measure( auto &&pos ) const {
    auto a = pos( std::array<PI,2>{ info.local_dirac_index, bounds[ 0 ].info.local_dirac_index } );
    auto b = pos( std::array<PI,2>{ info.local_dirac_index, bounds[ 1 ].info.local_dirac_index } );
    return b - a;
}

UTP TF DTP::measure() const {
    return bounds[ 1 ].vertex_pos[ 0 ] - bounds[ 0 ].vertex_pos[ 0 ];
}

// UTP void DTP::check_consistency( TF ) const {
// }

#undef UTP
#undef DTP

} // namespace sdot

template<class TF,class Arch>
std::ostream& operator<<( std::ostream &os, const sdot::Cell<TF,1,Arch> &p ) {
    for ( const auto &v : p.bounds )
        os << "\n  pos: " << v.vertex_pos << " dir: " << v.cut_dir << " dot: " << v.cut_dot << " ext: " << v.vertex_ext;
    return os;
}
