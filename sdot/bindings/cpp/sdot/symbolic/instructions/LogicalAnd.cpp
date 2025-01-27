#include <tl/support/string/to_string.h>
#include "LogicalAnd.h"
#include "Value.h"

namespace sdot {

RcPtr<Inst> LogicalAnd::from_operands( const RcPtr<Inst> &a, const RcPtr<Inst> &b ) {
    // known value ?
    if ( Opt<BigRational> va = a->constant_value() ) {
        if ( *va )
            return b;
        return a;
    }

    if ( Opt<BigRational> vb = b->constant_value() ) {
        if ( *vb )
            return a;
        return b;
    }

    // no simplification
    auto *res = new LogicalAnd;
    res->add_child( a );
    res->add_child( b );
    return res;
}

void LogicalAnd::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
}

RcPtr<Inst> LogicalAnd::clone( Vec<RcPtr<Inst>> &&new_children ) const {
    return from_operands( new_children[ 0 ], new_children[ 1 ] );
}

Str LogicalAnd::base_info() const {
    return "AndBoolean";
}

int LogicalAnd::type() const {
    return type_LogicalAnd;
}

}
