#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/Displayer.h>

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TCell>
class LocalWeightBounds {
public:
    using                Pt           = TCell::Pt;
    using                TF           = Pt::value_type;
    static constexpr int nb_dims      = Pt::ct_size;
 
    virtual bool         may_intersect( const TCell &test_cell, const TCell &domain ) const = 0;
    virtual TF           operator[]   ( PI index ) const = 0;
    virtual void         display      ( Displayer &ds ) const = 0;
};

} // namespace sdot
