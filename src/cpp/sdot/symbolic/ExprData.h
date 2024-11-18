#pragma once

#include <tl/support/memory/UniquePtr.h>
#include <tl/support/common_types.h>

namespace sdot {
class Inst;

/** */
class ExprData {
public:
    struct Val {
        virtual ~Val() {}
    };

    /**/ ExprData( PI num_in_list, const Inst *inst, Val *val ) : num_in_list( num_in_list ), inst( inst ), val{ val } {}

    PI             num_in_list;
    const Inst*    inst;
    UniquePtr<Val> val;
};

template<class T>
struct ExprVal : ExprData::Val {
    ExprVal( T &&val ) : val( std::move( val ) ) {}
    T val;
};

}
