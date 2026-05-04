#pragma once

#include <sdot/generated_includes/BatchOfCellWorkspace.h>
#include <sdot/generated_includes/PowerDiagram.h>
#include <sdot/generated_includes/BatchOfCell.h>

namespace sdot {

template<int ct_dim,class Arch,class TF,class TI>
struct PowerDiagramWorker {
    using Pt                   = DsVec<TF,ct_dim,Arch>; ///< point

    // inputs



    //
    void  for_each_point_in_bsp( auto &&func );
    void  for_each_cell        ( auto &&density, auto &&func );

    // attributes ---------------------------------------------------------------------------
    PowerDiagram<ct_dim,ct_dim,ct_dim,Arch,TF,TI> &power_diagram;
    BatchOfCellWorkspace<Arch,TF,TI> &cell_workspaces;
    BatchOfCell<ct_dim,Arch,TF,TI> &cells;
};

} // namespace sdot

#include "PowerDiagramWorker.cxx" // IWYU pragma: export
