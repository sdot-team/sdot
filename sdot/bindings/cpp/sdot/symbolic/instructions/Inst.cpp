#include <tl/support/compare.h>
#include <tl/support/ASSERT.h>
#include <tl/support/ERROR.h>
#include "Symbol.h"
#include "Value.h"
#include "Add.h"
#include "Mul.h"

namespace sdot {

Inst::~Inst() {
    for( PI i = 0; i < natural_args.size(); ++i )
        natural_args[ i ].second->parents.remove_first_unordered( this );
    for( PI i = 0; i < children.size(); ++i )
        children[ i ]->parents.remove_first_unordered( this );
}

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

void Inst::display( Displayer &ds ) const {
    Str res;
    display( res );
    ds << res;
}

bool Inst::operator<( const Inst &that ) const {
    return compare( that ) < 0;
}

void Inst::clear_natural_args() {
    for( PI i = 0; i < natural_args.size(); ++i )
        natural_args[ i ].second->parents.remove_first_unordered( this );
    natural_args.clear();
}

void Inst::add_natural_arg( Str name, const RcPtr<Inst> &inst ) {
    natural_args << Entry{ name, inst };
    inst->parents << this;
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

RcPtr<Inst> Inst::clone( Vec<RcPtr<Inst>> &&new_children ) const {
    ERROR( "this instruction should not be cloned" );
}

bool Inst::always_equal( const Inst &that ) const {
    return this == &that;
}

int Inst::compare( const Inst &b ) const {
    // compare the pointers
    const sdot::Inst &a = *this;
    if ( &a == &b )
        return 0;

    // compare the types
    int ta = a.type();
    int tb = b.type();
    if ( int c = ::compare( ta, tb ) )
        return c;

    if ( int c = ::compare( a.children.size(), b.children.size() ) )
        return c;
    for( PI n = 0; n < a.children.size(); ++n )
        if ( int c = ::compare( *a.children[ n ], *b.children[ n ] ) )
            return c;

    return compare_same( b );
}

int Inst::compare_same( const Inst &b ) const {
    return 0;
}

RcPtr<Inst> Inst::read_from( CompactReprReader &cr ) {
    TODO;
    return {};
}

void Inst::find_natural_args_rec( Vec<Vec<std::pair<Str,RcPtr<Inst>>>> &nargs ) {
    if ( natural_args.empty() ) {
        for( auto &ch : children )
            ch->find_natural_args_rec( nargs );
        return;
    }

    nargs.push_back_unique( natural_args );
}

RcPtr<Inst> Inst::apply( const Vec<std::pair<Str,RcPtr<Inst>>> &arguments ) {
    Vec<Vec<std::pair<Str,RcPtr<Inst>>>> nargs;
    find_natural_args_rec( nargs );
    if ( nargs.size() == 0 )
        throw NaturalArgsError{ "found no natural args for this expression" };
    if ( nargs.size() > 1 )
        throw NaturalArgsError{ "too mush natural args possibilities" };

    std::map<RcPtr<Inst>,RcPtr<Inst>> map;
    PI nb_unnamed = 0;
    for( const std::pair<Str,RcPtr<Inst>> &argument : arguments ) {
        if ( argument.first.empty() ) {
            if ( nb_unnamed >= nargs[ 0 ].size() )
                throw NaturalArgsError{ "too much natural args possibilities" };
            map[ nargs[ 0 ][ nb_unnamed++ ].second ] = argument.second;
        } else {
            SI index = nargs[ 0 ].index_first_checking( [&]( const auto &p ) { return p.first == argument.first; } );
            if ( index < 0 )
                throw NaturalArgsError{ "no natural args named '" + argument.first + "'" };
            map[ nargs[ 0 ][ index ].second ] = argument.second;
        }
    }

    return subs( map );
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

    return clone( std::move( new_children ) );
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
