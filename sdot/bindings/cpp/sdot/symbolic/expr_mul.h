#pragma once

#include "instructions/Value.h"
#include "instructions/Func.h"
#include <tl/support/TODO.h>
#include <tl/support/P.h>

namespace sdot {

inline RcPtr<Inst> expr_mul( const BigRational &ca, const RcPtr<Inst> &a, const BigRational &cb, const RcPtr<Inst> &b ) {
    if ( const auto *va = dynamic_cast<const Value *>( a.get() ) ) {
        if ( const auto *vb = dynamic_cast<const Value *>( b.get() ) ) {
            return Value::from_value( va->value.pow( ca ) * vb->value.pow( cb ) );
        }
    }

    return Func::from_operands( "mul", { a->pow_pair( ca ), b->pow_pair( cb ) } );
}

}