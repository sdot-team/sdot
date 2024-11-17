#include "sdot/symbolic/Inst.h"
#include <tl/support/compare.h>
#include "Symbol.h"
#include "Value.h"
#include "Func.h"

namespace sdot {

bool Inst::operator<( const Inst &that ) const {
    return compare( *this, that ) < 0;
}

void Inst::add_child( const RcPtr<Inst> &inst ) {
    children << inst;
}

std::pair<RcPtr<Inst>,BigRational> Inst::mul_pair() const {
    return { const_cast<Inst *>( this ), 1 }; // TODO: use CstRcPtr
}

}

int compare( const sdot::Inst &a, const sdot::Inst &b ) {
    int ta = a.type();
    int tb = b.type();

    if ( ta != tb )
        return ta - tb;

    if ( ta == sdot::Inst::type_Symbol ) {
        const auto *sa = static_cast<const sdot::Symbol *>( &a );
        const auto *sb = static_cast<const sdot::Symbol *>( &b );
        return compare( sa->name, sb->name );
    }

    if ( ta == sdot::Inst::type_Value ) {
        const auto *sa = static_cast<const sdot::Value *>( &a );
        const auto *sb = static_cast<const sdot::Value *>( &b );
        return compare( sa->value, sb->value );
    }

    if ( ta == sdot::Inst::type_Func ) {
        const auto *sa = static_cast<const sdot::Func *>( &a );
        const auto *sb = static_cast<const sdot::Func *>( &b );
        if ( int c = compare( sa->name, sb->name ) )
            return c;
        if ( int c = compare( sa->children, sb->children ) )
            return c;
        return compare( sa->coefficients, sa->coefficients );
    }

    TODO;
}
