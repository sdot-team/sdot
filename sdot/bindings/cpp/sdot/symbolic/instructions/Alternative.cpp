#include <tl/support/string/to_string.h>
#include "Alternative.h"
#include "Value.h"

namespace sdot {

RcPtr<Inst> Alternative::from_operands( const RcPtr<Inst> &index, const Vec<RcPtr<Inst>> &expr_list ) {
    // known index ?
    if ( Opt<BigRational> v = index->constant_value() )
        return expr_list[ PI( *v ) ];

    // if expr are all the same
    for( PI n = 1; ; ++n ) {
        if ( n == expr_list.size() )
            return expr_list[ 0 ];
        if ( ! expr_list[ 0 ]->always_equal( *expr_list[ n ] ) )
            break;
    }

    // no simplification
    auto *res = new Alternative;
    res->add_child( index );
    for( const auto &expr : expr_list)
        res->add_child( expr );
    return res;
}

void Alternative::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
}

RcPtr<Inst> Alternative::clone( Vec<RcPtr<Inst>> &&new_children ) const {
    return from_operands( new_children[ 0 ], { FromSizeAndIterator(), new_children.size() - 1, new_children.begin() + 1 } );
}

Str Alternative::base_info() const {
    return "Alternative";
}

int Alternative::type() const {
    return type_Alternative;
}

}
