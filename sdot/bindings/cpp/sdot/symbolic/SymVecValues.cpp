#include "ValueFromVec.h"

namespace sdot {

std::map<BigRational,RcPtr<Inst>> value_map;

Opt<BigRational> ValueFromVec::constant_value() const {
    return value;
}

bool ValueFromVec::always_equal( const Inst &that ) const {
    if ( auto *t = dynamic_cast<const Value *>( &that ) )
        return t->value == value;
    return false;
}

RcPtr<Inst> ValueFromVec::from_value( const BigRational &value ) {
    auto iter = value_map.find( value );
    if ( iter == value_map.end() ) {
        auto *v = new Value;
        v->value = value;

        iter = value_map.insert( iter, { value, v } );
    }
    return iter->second;
}

void ValueFromVec::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    cw.write_positive_int( type_Value, nb_types );
    cw << value;
}

void ValueFromVec::display( Displayer &ds ) const {
    ds << double( value );
    // ds << value;
}

RcPtr<Inst> ValueFromVec::subs( const std::map<Str,RcPtr<Inst>> &map ) const {
    return const_cast<Value *>( this );
}

}
