#pragma once

#include <tl/support/containers/CstSpanView.h>
#include <tl/support/containers/CstSpan.h>
#include <tl/support/memory/RcPtr.h>
#include <tl/support/Displayer.h>

namespace sdot {

/**
 * @brief 
 * 
 * @tparam T 
 */
template<class T>
class PoomVecInst {
public:
    using        CstSpanViewFunc      = void( CstSpanView<T> );

    virtual void get_values_by_chuncks( RcPtr<PoomVecInst> &ptr, CstSpanViewFunc func, PI beg, PI end ) = 0;
    virtual void get_values_by_chuncks( RcPtr<PoomVecInst> &ptr, CstSpanViewFunc func ) = 0;
    
    virtual void display              ( Displayer &ds ) const = 0; 
};

} // namespace sdot
