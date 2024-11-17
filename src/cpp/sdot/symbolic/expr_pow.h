#pragma once

#include <tl/support/TODO.h>
#include "Value.h"
#include "Func.h"

namespace sdot {

inline RcPtr<Inst> expr_pow( const RcPtr<Inst> &a, const RcPtr<Inst> &b ) {
    if ( const auto *va = dynamic_cast<const Value *>( a.get() ) ) {
        if ( const auto *vb = dynamic_cast<const Value *>( b.get() ) )
            return Value::from_value( pow( va->value, vb->value ) );
    }

    return Func::from_operands( "pow", { { a, 1 }, { b, 1 } } );
}

}
