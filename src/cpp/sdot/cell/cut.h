#pragma once

#include "CellWorker.h"

namespace sdot {

template<int ct_dim,class Arch,class TF,class TI>
void cut( Cell<ct_dim,Arch,TF,TI> &cell, CellWorkspace<Arch,TF,TI> &ws, const auto &cut_dir, auto cut_dot, SI cut_id ) {
    CellWorker<ct_dim,Arch,TF,TI> cw( cell, ws, cell.dim() );
    cw.cut( cut_dir, cut_dot, cut_id );
}

} // namespace sdot
