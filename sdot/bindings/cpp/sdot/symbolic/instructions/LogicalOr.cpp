#include <tl/support/string/to_string.h>
#include "LogicalOr.h"
#include "Value.h"
#include "sdot/symbolic/instructions/Inst.h"

namespace sdot {

RcPtr<Inst> LogicalOr::from_operands( const RcPtr<Inst> &a, const RcPtr<Inst> &b ) {
    // known value ?
    if ( Opt<BigRational> va = a->constant_value() ) {
        if ( *va )
            return a;
        return b;
    }

    if ( Opt<BigRational> vb = b->constant_value() ) {
        if ( *vb )
            return Value::from_value( 1 ); // HUM
        return a;
    }

    // no simplification
    auto *res = new LogicalOr;
    res->add_child( a );
    res->add_child( b );
    return res;
}

void LogicalOr::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
}

RcPtr<Inst> LogicalOr::clone( Vec<RcPtr<Inst>> &&new_children ) const {
    return from_operands( new_children[ 0 ], new_children[ 1 ] );
}

Str LogicalOr::base_info() const {
    return "LogicalOr";
}

int LogicalOr::type() const {
    return type_LogicalOr;
}

}
