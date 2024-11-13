#pragma once

#include "PoomVecInst_KV.h"
#include "PoomVec.h"

namespace sdot {

#define DTP template<class T>
#define UTP PoomVec<T>

DTP UTP::PoomVec( CstSpan<T> data ) {
    inst = new PoomVecInst_KV( data );
}

DTP void UTP::get_values_by_chuncks( const std::function<void( CstSpanView<T> )> & func, PI beg, PI end ) const {
    inst->get_values_by_chuncks( inst, func, beg, end );

}

DTP void UTP::get_values_by_chuncks( const std::function<void( CstSpanView<T> )> & func ) const {
    inst->get_values_by_chuncks( inst, func );
}

DTP PI UTP::size() const {
    return inst->size();
}

DTP void UTP::operator+=( const PoomVec<T> &that ) {
    // TODO: get_refs_...
    inst->operator+=( that );
}

DTP void UTP::operator-=( const PoomVec<T> &that ) {
    // TODO: get_refs_...
    inst->operator-=( that );
}

DTP void UTP::operator/=( const T &that ) {
    // TODO: get_refs_...
    inst->operator/=( that );
}

#undef DTP
#undef UTP

} // namespace sdot
