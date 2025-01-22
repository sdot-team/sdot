#include <limits>
#include <tl/support/string/to_string.h>
#include "Value.h"
#include "sdot/support/BigRational.h"

namespace sdot {

std::map<BigRational,RcPtr<Inst>> value_map;

bool Value::compare( const Value &a, const Value &b ) {
    return ::compare( a.value, b.value );
}

RcPtr<Inst> Value::from_value( const BigRational &value ) {
    auto iter = value_map.find( value );
    if ( iter == value_map.end() ) {
        auto *v = new Value;
        v->value = value;

        iter = value_map.insert( iter, { value, v } );
    }
    return iter->second;
}

Opt<BigRational> Value::constant_value() const {
    return value;
}

void Value::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    cw.write_positive_int( type_Value, nb_types );
    cw << value;
}

Str Value::base_info() const {
    return value.compact_repr();
}

}
