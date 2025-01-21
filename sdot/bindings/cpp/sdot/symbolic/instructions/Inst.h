#pragma once

#include <tl/support/string/CompactReprWriter.h>
#include <tl/support/containers/Vec.h>
#include <tl/support/containers/Opt.h>
#include <tl/support/memory/RcPtr.h>
#include <tl/support/Displayer.h>

#include "../../support/BigRational.h"
#include "../ExprData.h"

namespace sdot {

/** */
class Inst {
public:
    enum {              type_Value, type_ValueFromVec, type_Symbol, type_Func, type_SymbolicArray, nb_types };
   
    virtual            ~Inst          () {}
         
    virtual auto        constant_value() const -> Opt<BigRational>;
    virtual bool        always_equal  ( const Inst &that ) const;
    bool                operator<     ( const Inst &that ) const;
    virtual auto        mul_pair      ( const BigRational &coeff ) const -> std::pair<RcPtr<Inst>,BigRational>;
    virtual auto        pow_pair      ( const BigRational &coeff ) const -> std::pair<RcPtr<Inst>,BigRational>;
    virtual void        display       ( Displayer &ds ) const = 0;
    virtual int         type          () const = 0;
    virtual RcPtr<Inst> subs          ( const std::map<Str,RcPtr<Inst>> &map ) const = 0;
      
    virtual void        ct_rt_split   ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const = 0;
    static RcPtr<Inst>  read_from     ( CompactReprReader &cr );
      
    void                add_child     ( const RcPtr<Inst> &inst );
    int                 compare       ( const Inst &that ) const;
   
    static PI           rt_data_num   ( Vec<ExprData> &data_map, const Inst *inst, const std::function<ExprData::Val *()> &make_rt_data );
   
    PI                  ref_count = 0;
    Vec<RcPtr<Inst>>    children;
    Vec<Inst*>          parents;
};


}


