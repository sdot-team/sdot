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

Str Value::ct_rt_split( Vec<std::pair<const Inst *,ExprData>> &data_map ) const {
    return value.compact_str_repr();
}

void Value::display( Displayer &ds ) const {
    ds << double( value );
}

}
