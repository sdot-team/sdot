#include <sdot/generated_includes/BatchOfCellWorkspace.h>
#include <sdot/generated_includes/PowerDiagram.h>
#include <sdot/generated_includes/BatchOfCell.h>
#include "../Bsp/for_each_point_in_bsp.h"
#include "../Cell/make_aligned_simplex.h"
#include "../Cell/cut.h"

#include "../Distribution/PolynomialGridWorker.h"

// #include "../support/P.h"

namespace sdot {

template<int ct_dim,typename Arch,typename TF,typename TI>
void for_each_cell( const PowerDiagram<ct_dim,ct_dim,ct_dim,Arch,TF,TI> &pd, BatchOfCell<ct_dim,Arch,TF,TI> &cells, BatchOfCellWorkspace<Arch,TF,TI> &cell_workspaces, auto &&density, auto &&func ) {
    using Pt = DsVec<TF,ct_dim,Arch>;

    CellWorkspace<Arch,TF,TI> cell_workspace = cell_workspaces.row( 0 );
    Cell<ct_dim,Arch,TF,TI> cell = cells.row( 0 );

    for_each_point_in_bsp( pd.bsp, [&]( auto i0 ) {
        // make_aligned_simplex( cell, CellBoundary::INFINITE, CtInt<ct_dim>() );
        density.init_cell( cell );

        const Pt p0 = pd.positions.row( i0 );
        const TF w0 = pd.weights( i0 );
        for_each_point_in_bsp( pd.bsp, [&]( auto i1 ) {
            if ( i1 == i0 )
                return;
            const Pt p1 = pd.positions.row( i1 );
            const TF w1 = pd.weights( i1 );

            const Pt dir = p1 - p0;

            auto n = norm_2_p2( dir );
            auto s0 = dot( dir, p0 );
            auto s1 = dot( dir, p1 );

            auto off = s0 + ( 1 + ( w0 - w1 ) / n ) / 2 * ( s1 - s0 );

            cut( cell, cell_workspace, dir, off, i1 );
        } );

        func( cell, 0 );
    } );
}

template<int ct_dim,typename Arch,typename TF,typename TI>
void for_each_cell( const PowerDiagram<ct_dim,ct_dim,ct_dim,Arch,TF,TI> &pd, BatchOfCell<ct_dim,Arch,TF,TI> &cells, BatchOfCellWorkspace<Arch,TF,TI> &cell_workspaces, auto &&func ) {
    // make_aligned_simplex( cell, CellBoundary::INFINITE, CtInt<ct_dim>() );
    TODO;
}

} // namespace sdot
