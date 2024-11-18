#pragma once

#include <tl/support/memory/UniquePtr.h>
#include <tl/support/common_types.h>

namespace sdot {
class Inst;

/** */
class ExprData {
public:
    struct Val {
        virtual Val *clone() const = 0;
        virtual ~Val() {}
    };

    /**/ ExprData( PI num_in_list, const Inst *inst, Val *val ) : num_in_list( num_in_list ), inst( inst ), val{ val } {}
    /**/ ExprData( const ExprData &that ) : num_in_list( that.num_in_list ), inst( that.inst ), val{ that.val->clone() } {}
    /**/ ExprData( ExprData &&that ) : num_in_list( that.num_in_list ), inst( that.inst ), val{ std::move( that.val ) } {}

    PI             num_in_list;
    const Inst*    inst;
    UniquePtr<Val> val;
};

template<class T>
struct ExprVal : ExprData::Val {
    ExprVal( T &&val ) : val( std::move( val ) ) {}
    virtual ExprData::Val *clone() const { return new ExprVal( T( val ) ); }
    T val;
};

}
