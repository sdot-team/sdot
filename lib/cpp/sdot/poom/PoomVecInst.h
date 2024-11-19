#pragma once

#include <tl/support/containers/CstSpanView.h>
#include <tl/support/containers/CstSpan.h>
#include <tl/support/memory/RcPtr.h>
#include <tl/support/Displayer.h>

namespace sdot {
template<class T>
class PoomVec;

/**
 * @brief 
 * 
 * @tparam T 
 */
template<class T>
class PoomVecInst {
public:
    virtual     ~PoomVecInst          () {}

    virtual void get_values_by_chuncks( RcPtr<PoomVecInst> &ptr, const std::function<void( CstSpanView<T> )> &func, PI beg, PI end ) const = 0;
    virtual void get_values_by_chuncks( RcPtr<PoomVecInst> &ptr, const std::function<void( CstSpanView<T> )> &func ) const = 0;

    virtual void display              ( Displayer &ds ) const = 0; 
    virtual PI   size                 () const = 0;
    
    virtual void operator+=           ( const PoomVec<T> &that ) = 0;
    virtual void operator-=           ( const PoomVec<T> &that ) = 0;
    virtual void operator/=           ( const T &that ) = 0;

    PI           ref_count            = 0;
};

} // namespace sdot
