#pragma once

#include <tl/support/containers/Vec.h>
#include "PoomVec.h"

namespace sdot {

/**
 * @brief 
 * 
 * @tparam T 
 */
template<class T>
class PoomVecInst_KV : public PoomVecInst<T> {
public:
    /* */        PoomVecInst_KV       ( CstSpan<T> data ) : data( data ) {}

    virtual void get_values_by_chuncks( RcPtr<PoomVecInst<T>> &ptr, const std::function<void( CstSpanView<T> )> &func, PI beg, PI end ) const { func( { data.data(), beg, end, data.size() } ); }
    virtual void get_values_by_chuncks( RcPtr<PoomVecInst<T>> &ptr, const std::function<void( CstSpanView<T> )> &func ) const { func( { data.data(), 0, data.size(), data.size() } ); }
    
    virtual void display              ( Displayer &ds ) const { ds << data; }
    virtual PI   size                 () const { return data.size(); }

    virtual void operator+=           ( const PoomVec<T> &that ) { that.get_values_by_chuncks( [&]( CstSpanView<T> that ) { for( PI i = that.beg_index(); i < that.end_index(); ++i ) data[ i ] += that[ i ]; } ); }
    virtual void operator-=           ( const PoomVec<T> &that ) { that.get_values_by_chuncks( [&]( CstSpanView<T> that ) { for( PI i = that.beg_index(); i < that.end_index(); ++i ) data[ i ] -= that[ i ]; } ); }
    virtual void operator/=           ( const T &that ) { for( PI i = 0; i < data.size(); ++i ) data[ i ] /= that; }

    Vec<T>       data;
};

} // namespace sdot
