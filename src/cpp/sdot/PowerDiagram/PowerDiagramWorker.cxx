#pragma once

#include "PowerDiagramWorker.h"
#include "../Cell/CellWorker.h"

namespace sdot {

#define UTP template<int ct_dim,class Arch,class TF,class TI>
#define DTP PowerDiagramWorker<ct_dim,Arch,TF,TI>

UTP DTP::Pt DTP::position( PI n ) const {
    return power_diagram.positions.row( n );
}

UTP int DTP::for_each_point_in_bsp( auto &&func ) {
    for( PI i = 0; i < power_diagram.bsp.sorted_vertex_indices.size(); ++i )
        if ( int error = func( power_diagram.bsp.sorted_vertex_indices[ i ] ) )
            return error;
    return 0;
}

UTP int DTP::for_each_cell( auto &&density, auto &&func ) {
    using Pt = Vector<TF,Arch,ct_dim,Arch>;

    CellWorkspace<Arch,TF,TI> cell_workspace = cell_workspaces.row( 0 );
    Cell<ct_dim,Arch,TF,TI> cell = cells.row( 0 );
    CellWorker<ct_dim,Arch,TF,TI> cell_worker( cell, cell_workspace, cell.dim() );

    return for_each_point_in_bsp( [&]( auto i0 ) {
        // make_aligned_simplex( cell, CellBoundary::INFINITE, CtInt<ct_dim>() );
        density.init_cell( cell );

        const Pt p0 = power_diagram.positions.row( i0 );
        const TF w0 = power_diagram.weights( i0 );
        for_each_point_in_bsp( [&]( auto i1 ) {
            if ( i1 == i0 )
                return 0;

            const Pt p1 = power_diagram.positions.row( i1 );
            const TF w1 = power_diagram.weights( i1 );

            const Pt dir = p1 - p0;

            auto n = norm_2_p2( dir );
            auto s0 = dot( dir, p0 );
            auto s1 = dot( dir, p1 );

            auto off = s0 + ( 1 + ( w0 - w1 ) / n ) / 2 * ( s1 - s0 );

            cell_worker.cut( dir, off, i1 );
            return 0;
        } );

        return func( cell_worker, 0, i0 );
    } );
}

#undef UTP
#undef DTP

} // namespace sdot
