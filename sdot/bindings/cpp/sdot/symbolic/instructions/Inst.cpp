#include <tl/support/compare.h>
#include "Symbol.h"
#include "Value.h"
#include "Func.h"

namespace sdot {

bool Inst::operator<( const Inst &that ) const {
    return compare( that ) < 0;
}

void Inst::add_child( const RcPtr<Inst> &inst ) {
    children << inst;
}

std::pair<RcPtr<Inst>,BigRational> Inst::mul_pair( const BigRational &coeff ) const {
    return { const_cast<Inst *>( this ), coeff }; // TODO: use CstRcPtr
}

std::pair<RcPtr<Inst>,BigRational> Inst::pow_pair( const BigRational &coeff ) const {
    return { const_cast<Inst *>( this ), coeff }; // TODO: use CstRcPtr
}

Opt<BigRational> Inst::constant_value() const {
    return {};
}

bool Inst::always_equal( const Inst &that ) const {
    return this == &that;
}

int Inst::compare( const sdot::Inst &b ) const {
    const sdot::Inst &a = *this;

    int ta = a.type();
    int tb = b.type();

    if ( ta != tb )
        return ta - tb;

    if ( ta == sdot::Inst::type_Symbol ) {
        const auto *sa = static_cast<const sdot::Symbol *>( &a );
        const auto *sb = static_cast<const sdot::Symbol *>( &b );
        return ::compare( sa->name, sb->name );
    }

    if ( ta == sdot::Inst::type_Value ) {
        const auto *sa = static_cast<const sdot::Value *>( &a );
        const auto *sb = static_cast<const sdot::Value *>( &b );
        return ::compare( sa->value, sb->value );
    }

    if ( ta == sdot::Inst::type_Func ) {
        const auto *sa = static_cast<const sdot::Func *>( &a );
        const auto *sb = static_cast<const sdot::Func *>( &b );
        if ( int c = ::compare( sa->name, sb->name ) )
            return c;
        if ( int c = ::compare( sa->children, sb->children ) )
            return c;
        return ::compare( sa->coefficients, sa->coefficients );
    }

    TODO;
    return 0;
}

RcPtr<Inst> Inst::read_from( CompactReprReader &cr ) {
    TODO;
    return {};
}

PI Inst::rt_data_num( Vec<ExprData> &data_map, const Inst *inst, const std::function<ExprData::Val *()> &make_rt_data ) {
    for( PI i = 0; i < data_map.size(); ++i )
        if ( data_map[ i ].inst == inst )
            return i;
    PI res = data_map.size();
    data_map.push_back( res, inst, make_rt_data() );
    return res;
}

}
