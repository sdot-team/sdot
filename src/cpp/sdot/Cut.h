#pragma once

#include "sdot/CutType.h"
#include <tl/support/containers/Vec.h>
#include <tl/support/Displayer.h>

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TF,int nb_dims,class CutInfo>
class Cut {
public:
    using    Pt     = Vec<TF,nb_dims>;
 
    /**/     Cut    ( Pt dir_td, CutInfo &&info, const Pt &dir, TF off ) : _dir_td( dir_td ), info( std::move( info ) ), dir( dir ), off( off ) {}
    /**/     Cut    ( CutInfo &&info, const Pt &dir, TF off ) : info( std::move( info ) ), dir( dir ), off( off ) {}
    /**/     Cut    () {}

    T_i TF   dir_td ( CtInt<i> td, auto ind ) const { if constexpr ( i < nb_dims ) return _dir_td[ ind ]; else return dir[ ind ]; }
    T_i auto dir_td ( CtInt<i> td ) const { if constexpr ( i < nb_dims ) return _dir_td.slice( CtInt<0>(), td ); else return dir; }

    void     display( Displayer &ds ) const { bool b = info.type == CutType::Boundary; DS_OJBECT( dir, off, _dir_td, info.i1, info.type ); }

    Pt       _dir_td; ///< dir used if true_dimensionality < nb_dims
    CutInfo  info;
    Pt       dir;
    TF       off;
};

} // namespace sdot
