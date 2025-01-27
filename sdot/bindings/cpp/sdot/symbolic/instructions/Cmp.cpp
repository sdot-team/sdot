#include "Value.h"
#include "Cmp.h"

namespace sdot {

int Cmp::compare_same( const Inst &that ) const {
    const auto &b = static_cast<const Cmp &>( that );
    return ::compare( (int)cmp_type, (int)b.cmp_type );
}

RcPtr<Inst> Cmp::from_operands( CmpType cmp_type, const RcPtr<Inst> &a ) {
    // all known ?
    if ( Opt<BigRational> pa = a->constant_value() ) {
        switch ( cmp_type ) {
        case CmpType::Equal: return Value::from_value( *pa == 0 );
        case CmpType::InfEq: return Value::from_value( *pa <= 0 );
        case CmpType::SupEq: return Value::from_value( *pa >= 0 );
        case CmpType::Neq:   return Value::from_value( *pa != 0 );
        case CmpType::Inf:   return Value::from_value( *pa < 0 );
        case CmpType::Sup:   return Value::from_value( *pa > 0 );
        }
    }

    // else
    auto *res = new Cmp;
    res->cmp_type = cmp_type;
    res->add_child( a );
    return res;
}

void Cmp::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
}

void Cmp::display( Str &res, int prio ) const {
    if ( prio > prio_Cmp )
        res += "( ";

    children[ 0 ]->display( res, prio_Cmp );
    res += " " + base_info();

    if ( prio > prio_Cmp )
        res += " )";
}

RcPtr<Inst> Cmp::clone( Vec<RcPtr<Inst>> &&new_children ) const {
    return from_operands( cmp_type, new_children[ 0 ] );
}

Str Cmp::base_info() const {
    switch ( cmp_type ) {
    case CmpType::Equal: return "== 0";
    case CmpType::InfEq: return "<= 0";
    case CmpType::SupEq: return ">= 0";
    case CmpType::Neq:   return "!= 0";
    case CmpType::Inf:   return "< 0";
    case CmpType::Sup:   return "> 0";
    }
}

int Cmp::type() const {
    return type_Cmp;
}

}
