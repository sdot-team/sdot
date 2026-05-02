#pragma once

#include <sdot/generated_includes/Bsp.h>

namespace sdot {

template<class Arch,class TI,class TF,int ct_dim>
void for_each_point_in_bsp( const Bsp<ct_dim,Arch,TF,TI> &bsp, auto &&func ) {
    for( PI i = 0; i < bsp.sorted_vertex_indices.size(); ++i )
        func( bsp.sorted_vertex_indices[ i ] );
}


} // namespace sdot
