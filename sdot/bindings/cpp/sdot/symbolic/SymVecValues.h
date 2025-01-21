#pragma once

#include <tl/support/memory/RcPtr.h>
#include <tl/support/containers/Vec.h>
#include <tl/support/Displayer.h>

namespace sdot {
class Inst;

/** */
class SymVecValues {
public:
    virtual Str  item_type() const;
    virtual void display  ( Displayer &ds ) const;

    PI           ref_count = 0;
    Vec<Inst*>   parents;
};

/** */
template<class Item>
class SymVecValues_ : public SymVecValues {
public:
    virtual Str  item_type() const;
    virtual void display  ( Displayer &ds ) const;

    Vec<Item>    items;
};

}
