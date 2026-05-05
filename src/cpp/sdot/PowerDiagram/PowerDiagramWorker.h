#pragma once

#include <sdot/generated_includes/BatchOfCellWorkspace.h>
#include <sdot/generated_includes/PowerDiagram.h>
#include <sdot/generated_includes/BatchOfCell.h>

namespace sdot {

template<int ct_dim,class Arch,class TF,class TI>
struct PowerDiagramWorker {
    using Pt                                 = DsVec<TF,ct_dim,Arch>; ///< point

    // inputs
    Pt    position                           ( PI n ) const;

    //
    int   for_each_point_in_bsp              ( auto &&func );
    int   for_each_cell                      ( auto &&density, auto &&func ); ///< func( cell_worker, batch_index, point_index ) -> bool to continue;

    // attributes ---------------------------------------------------------------------------
    PowerDiagram<ct_dim,ct_dim,ct_dim,Arch,TF,TI> &power_diagram;
    BatchOfCellWorkspace<Arch,TF,TI> &cell_workspaces;
    BatchOfCell<ct_dim,Arch,TF,TI> &cells;
};

} // namespace sdot

#include "PowerDiagramWorker.cxx" // IWYU pragma: export
