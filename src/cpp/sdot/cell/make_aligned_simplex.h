#include <sdot/generated_includes/Cell.h>
#include "CellBoundary.h"

#define UTP template<int ct_dim,class Arch,class TF,class TI>
#define DTP Cell<ct_dim,Arch,TF,TI>

namespace sdot {

UTP void make_aligned_simplex( DTP &cell, SI cut_id, CtInt<ct_dim> ) {
}

void make_empty_cell( auto &&p ) {
    make_aligned_simplex( p.cell, p.cell.INFINITE, p.ct_dim );
}

void make_empty_cell_backward( auto && ) {
    TODO; // make_aligned_simplex( p.cell, p.cell.INFINITE );
}

#undef UTP
#undef DTP

} // namespace sdot
