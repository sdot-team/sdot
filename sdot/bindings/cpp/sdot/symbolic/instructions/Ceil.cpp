#include "Value.h"
#include "Ceil.h"

namespace sdot {

RcPtr<Inst> Ceil::from_operands( const RcPtr<Inst> &a ) {
    //
    if ( Opt<BigRational> v = a->constant_value() )
        return Value::from_value( ceil( *v ) );
    
    //
    auto *res = new Ceil;
    res->add_child( a );
    return res;
}

void Ceil::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
}

RcPtr<Inst> Ceil::clone( Vec<RcPtr<Inst>> &&new_children ) const {
    return from_operands( std::move( new_children[ 0 ] ) );
}

Str Ceil::base_info() const {
    return "Ceil";
}

int Ceil::type() const {
    return type_Ceil;
}

}
