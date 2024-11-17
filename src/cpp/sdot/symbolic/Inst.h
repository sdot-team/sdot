#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/memory/RcPtr.h>
#include <tl/support/Displayer.h>

#include "../support/BigRational.h"
#include "ExprData.h"

namespace sdot {

/** */
class Inst {
public:
    enum {           type_Value, type_Symbol, type_Func };

    virtual         ~Inst       () {}
      
    bool             operator<  ( const Inst &that ) const;
    virtual auto     mul_pair   ( const BigRational &coeff ) const -> std::pair<RcPtr<Inst>,BigRational>;
    virtual auto     pow_pair   ( const BigRational &coeff ) const -> std::pair<RcPtr<Inst>,BigRational>;
    virtual void     display    ( Displayer &ds ) const = 0;
    virtual int      type       () const = 0;

    Str              ct_rt_split( Vec<std::pair<const Inst *,ExprData>> &data_map ) const;
  
    void             add_child  ( const RcPtr<Inst> &inst );
    int              compare    ( const Inst &that ) const;

    PI               ref_count = 0;
    Vec<RcPtr<Inst>> children;
};


}


