#pragma once

#include <tl/support/memory/UniquePtr.h>

namespace sdot {

/** */
class ExprData {
public:
    struct Val {
        virtual ~Val() {}
    };

    UniquePtr<Val> val;
};

template<class T>
struct ExprVal : ExprData::Val {
};

}
