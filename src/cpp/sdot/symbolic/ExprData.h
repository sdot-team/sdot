#pragma once

#include <tl/support/memory/UniquePtr.h>

namespace sdot {

/** */
class ExprData {
public:
    struct Val {
        virtual ~Val() {}
    };

    ExprData( Val *val ) : val{ val } {}

    UniquePtr<Val> val;
};

template<class T>
struct ExprVal : ExprData::Val {
    ExprVal( T &&val ) : val( std::move( val ) ) {}
    T val;
};

}
