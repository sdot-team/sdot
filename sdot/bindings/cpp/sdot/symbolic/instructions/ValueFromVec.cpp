#include "ValueFromVec.h"

namespace sdot {

RcPtr<Inst> ValueFromVec::from_vec( const RcPtr<SymVecValues> &vec, const RcPtr<Inst> &ind ) {
    TODO;
}

bool ValueFromVec::always_equal( const Inst &that ) const {
    TODO;
    return false;
}

void ValueFromVec::ct_rt_split( CompactReprWriter &cw, Vec<ExprData> &data_map ) const {
    TODO;
    // cw.write_positive_int( type_Value, nb_types );
    // cw << value;
}

void ValueFromVec::display( Displayer &ds ) const {
    // ds.set_next_type( "ValueFromVec" );
    // ds.start_object()
    // ds << "ValueFromVec";
    // ds << value;
    DS_OBJECT( ValueFromVec, vec, children );
}

RcPtr<Inst> ValueFromVec::subs( const std::map<Str,RcPtr<Inst>> &map ) const {
    TODO;
    //return const_cast<Value *>( this );
}

}
