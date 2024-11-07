#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/Displayer.h>

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TF,int nb_dims>
class AffineTransformation {
public:
    using       LT        = Vec<Vec<TF,nb_dims>,nb_dims>; ///< 
    using       Pt        = Vec<TF,nb_dims>; ///< 

    Pt          operator()( const Pt &p ) const { return sp( linear_transformation, p ) + translation; }
    void        display   ( Displayer &ds ) const { DS_OJBECT( linear_transformation, translation ); }

    LT          linear_transformation; ///<
    Pt          translation;           ///<
};

} // namespace sdot
