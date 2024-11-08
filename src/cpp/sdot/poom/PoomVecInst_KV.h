#pragma once

#include <tl/support/containers/Vec.h>
#include "PoomVecInst.h"

namespace sdot {

/**
 * @brief 
 * 
 * @tparam T 
 */
template<class T>
class PoomVecInst_KV : public PoomVecInst<T> {
public:
    using        CstSpanViewFunc      = PoomVecInst<T>::CstSpanViewFunc;

    /* */        PoomVecInst_KV       ( CstSpan<T> data ) : data( data ) {}

    virtual void get_values_by_chuncks( RcPtr<PoomVecInst<T>> &ptr, CstSpanViewFunc func, PI beg, PI end ) { func( { data.data(), beg, end, data.size() } ); }
    virtual void get_values_by_chuncks( RcPtr<PoomVecInst<T>> &ptr, CstSpanViewFunc func ) { func( { data.data(), 0, data.size(), data.size() } ); }
    
    virtual void display              ( Displayer &ds ) const { ds << data; }

    Vec<T>       data;
};

} // namespace sdot
