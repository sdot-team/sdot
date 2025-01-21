#include <tl/support/compare.h>
#include <tl/support/ASSERT.h>
#include <tl/support/ERROR.h>
#include "Symbol.h"
#include "Value.h"

namespace sdot {

void Inst::display( Str &res, int prio ) const {
    res += base_info();
    if ( children.size() ) {
        res += "(";
        for( PI i = 0; i < children.size(); ++i ) {
            res += ( i ? "," : "" );
            children[ i ]->display( res );
        }
        res += ")";
    }
}

bool Inst::operator<( const Inst &that ) const {
    return compare( that ) < 0;
}

void Inst::add_child( const RcPtr<Inst> &inst ) {
    inst->parents << this;
    children << inst;
}

std::pair<BigRational,RcPtr<Inst>> Inst::mul_pair( const BigRational &coeff ) const {
    return { coeff, const_cast<Inst *>( this ) }; // TODO: use CstRcPtr
}

std::pair<BigRational,RcPtr<Inst>> Inst::pow_pair( const BigRational &coeff ) const {
    return { coeff, const_cast<Inst *>( this ) }; // TODO: use CstRcPtr
}

Opt<BigRational> Inst::constant_value() const {
    return {};
}

RcPtr<Inst> Inst::clone( const Vec<RcPtr<Inst>> &new_children ) const {
    ERROR( "this instruction should not be cloned" );
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

    // if ( ta == sdot::Inst::type_Func ) {
    //     const auto *sa = static_cast<const sdot::Func *>( &a );
    //     const auto *sb = static_cast<const sdot::Func *>( &b );
    //     if ( int c = ::compare( sa->name, sb->name ) )
    //         return c;
    //     if ( int c = ::compare( sa->children, sb->children ) )
    //         return c;
    //     return ::compare( sa->coefficients, sa->coefficients );
    // }

    TODO;
    return 0;
}

RcPtr<Inst> Inst::read_from( CompactReprReader &cr ) {
    TODO;
    return {};
}

RcPtr<Inst> Inst::subs( const std::map<RcPtr<Inst>,RcPtr<Inst>> &map ) {
    auto iter = map.find( this );
    if ( iter != map.end() )
        return iter->second;

    Vec<RcPtr<Inst>> new_children( FromReservationSize(), children.size() );
    bool all_the_same = true;
    for( const RcPtr<Inst> &ch : children ) {
        RcPtr<Inst> nch = ch->subs( map );
        all_the_same &= ( ch == nch );
        new_children << nch;
    }

    if ( all_the_same )
        return this;

    return clone( new_children );
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
