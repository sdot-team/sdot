#pragma once

#include "PoomVecInst_KV.h"
#include "PoomVec.h"

namespace sdot {

#define DTP template<class T>
#define UTP PoomVec<T>

DTP UTP::PoomVec( CstSpan<T> data ) {
    inst = new PoomVecInst_KV( data );
}

DTP void UTP::get_values_by_chuncks( CstSpanViewFunc func, PI beg, PI end ) {
    inst->get_values_by_chuncks( inst, func, beg, end );

}

DTP void UTP::get_values_by_chuncks( CstSpanViewFunc func ) {
    inst->get_values_by_chuncks( inst, func );
}

#undef DTP
#undef UTP

} // namespace sdot
