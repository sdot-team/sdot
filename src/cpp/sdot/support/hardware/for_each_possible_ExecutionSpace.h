#pragma once

#ifdef __CUDACC__
    #include "ExecutionSpace_Cuda.h"
#endif
#include "ExecutionSpace_Cpu.h"
#include "../Ct.h"

namespace sdot {

void for_each_possible_ExecutionSpace( auto &&func ) {
    #ifdef __CUDACC__
        func( ExecutionSpace_Cuda() );
    #endif
    func( ExecutionSpace_Cpu() );
}

auto nb_possible_ExecutionSpace() {
    #ifdef __CUDACC__
        return Ct<int,2>();
    #else
        return Ct<int,1>();
    #endif
}

} // namespace sdot
