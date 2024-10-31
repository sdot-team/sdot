#pragma once

#include <tl/support/containers/Vec.h>

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TF,int nb_dims,class CutInfo>
class Cut {
public:
    using    Ps    = Vec<TF,nb_dims-1>;
    using    Pt    = Vec<TF,nb_dims>;

    T_i TF   dir_td( CtInt<i> td, auto ind ) const { if constexpr ( i < nb_dims ) return _dir_td[ ind ]; else return dir[ ind ]; }
    T_i auto dir_td( CtInt<i> td ) const { if constexpr ( i < nb_dims ) return _dir_td.slice( CtInt<0>(), td ); else return dir; }
    T_i TF   off_td( CtInt<i> td ) const { if constexpr ( i < nb_dims ) return _off_td; else return off; }

    CutInfo  info;
    Pt       dir;
    TF       off;

    Ps       _dir_td; ///< dir used if true_dimensionality < nb_dims
    TF       _off_td; ///< off used if true_dimensionality < nb_dims
};

} // namespace sdot
