#include "Value.h"
#include "Frac.h"

namespace sdot {

RcPtr<Inst> Frac::from_operands( const RcPtr<Inst> &a ) {
    //
    if ( Opt<BigRational> v = a->constant_value() )
        return Value::from_value( frac( *v ) );
    
    //
    auto *res = new Frac;
    res->add_child( a );
    return res;
}

void Frac::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
}

RcPtr<Inst> Frac::clone( Vec<RcPtr<Inst>> &&new_children ) const {
    return from_operands( std::move( new_children[ 0 ] ) );
}

Str Frac::base_info() const {
    return "Frac";
}

int Frac::type() const {
    return type_Frac;
}

}
