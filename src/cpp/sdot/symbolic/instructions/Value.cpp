#include "Value.h"

namespace sdot {

std::map<BigRational,RcPtr<Inst>> value_map;

RcPtr<Inst> Value::from_value( const BigRational &value ) {
    auto iter = value_map.find( value );
    if ( iter == value_map.end() ) {
        auto *v = new Value;
        v->value = value;

        iter = value_map.insert( iter, { value, v } );
    }
    return iter->second;
}

void Value::ct_rt_split( CompactReprWriter &cw, Vec<std::pair<const Inst *,ExprData>> &data_map ) const {
    cw.write_positive_int( type_Value, nb_types );
    cw << value;
}

void Value::display( Displayer &ds ) const {
    ds << double( value );
    // ds << value;
}

RcPtr<Inst> Value::subs( const std::map<Str,RcPtr<Inst>> &map ) const {
    return const_cast<Value *>( this );
}

}
