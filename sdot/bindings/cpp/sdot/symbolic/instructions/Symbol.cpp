#include <tl/support/ERROR.h>
#include "Symbol.h"

namespace sdot {

RcPtr<Inst> Symbol::from_name( const Str &name ) {
    auto *v = new Symbol;
    v->name = name;
    return v;
}

void Symbol::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    cw.write_positive_int( type_Symbol, nb_types );
    cw << name;
}

Str Symbol::base_info() const {
    return name;
}

RcPtr<Inst> Symbol::clone( const Vec<RcPtr<Inst>> &new_children ) const {
    ERROR( "cannot be cloned" );
}

}
