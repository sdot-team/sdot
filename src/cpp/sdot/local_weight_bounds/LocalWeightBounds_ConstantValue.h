#pragma once

#include "LocalWeightBounds.h"

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TCell>
class LocalWeightBounds_ConstantValue : public LocalWeightBounds<TCell> {
public:
    using        TP                             = LocalWeightBounds<TCell>;
    using        TF                             = TP::TF;

    /* */        LocalWeightBounds_ConstantValue( TF value ) : value( value ) {}

    virtual bool may_intersect                  ( const TCell &test_cell, const TCell &domain ) const {
        TODO;
    }
    
    virtual TF   operator[]                     ( PI index ) const { return value; }

    virtual void display                        ( Displayer &ds ) const { DS_OJBECT( value ); }

    TF           value;
};

} // namespace sdot
