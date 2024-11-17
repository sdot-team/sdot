#pragma once

#include <tl/support/TODO.h>
#include "Value.h"
#include "Func.h"

namespace sdot {

inline RcPtr<Inst> add( const RcPtr<Inst> &a, const RcPtr<Inst> &b ) {
    if ( const auto *va = dynamic_cast<const Value *>( a.get() ) ) {
        if ( const auto *vb = dynamic_cast<const Value *>( b.get() ) )
            return Value::from_value( va->value + vb->value );
    }

    return Func::from_operands( "add", { a->mul_pair(), b->mul_pair() } );
}

}