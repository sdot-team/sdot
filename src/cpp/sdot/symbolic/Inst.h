#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/memory/RcPtr.h>
#include <tl/support/Displayer.h>

#include "../support/BigRational.h"

namespace sdot {

/** */
class Inst {
public:
    enum {           type_Symbol, type_Value, type_Func };

    virtual         ~Inst     () {}
    
    bool             operator<( const Inst &that ) const;
    virtual auto     mul_pair ( const BigRational &coeff ) const -> std::pair<RcPtr<Inst>,BigRational>;
    virtual auto     pow_pair ( const BigRational &coeff ) const -> std::pair<RcPtr<Inst>,BigRational>;
    virtual void     display  ( Displayer &ds ) const = 0;
    virtual int      type     () const = 0;

    void             add_child( const RcPtr<Inst> &inst );

    PI               ref_count = 0;
    Vec<RcPtr<Inst>> children;
};

}

int compare( const sdot::Inst &a, const sdot::Inst &b );
