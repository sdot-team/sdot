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
    enum {              type_Value, type_Symbol, type_Array, type_Add, type_Mul, type_Cmp, type_Alternative, type_Ceil, type_Frac, type_LogicalAnd, type_LogicalOr, nb_types };
    enum {              prio_Cmp, prio_Add, prio_Mul, prio_Pow };
    struct              NaturalArgsError     { Str msg; };
    using               Entry                = std::pair<Str,RcPtr<Inst>>;
           
    virtual            ~Inst                 ();
           
    virtual auto        constant_value       () const -> Opt<BigRational>;
    virtual int         compare_same         ( const Inst &that ) const;
    virtual bool        always_equal         ( const Inst &that ) const;
    bool                operator<            ( const Inst &that ) const;
    virtual Str         base_info            () const = 0;
    virtual auto        mul_pair             ( const BigRational &coeff ) const -> std::pair<BigRational,RcPtr<Inst>>;
    virtual auto        pow_pair             ( const BigRational &coeff ) const -> std::pair<BigRational,RcPtr<Inst>>;
    virtual void        display              ( Str &res, int prio = 0 ) const;
    virtual void        display              ( Displayer &ds ) const;
    virtual RcPtr<Inst> clone                ( Vec<RcPtr<Inst>> &&new_children ) const;
    virtual int         type                 () const = 0;
    
    RcPtr<Inst>         apply                ( const Vec<std::pair<Str,RcPtr<Inst>>> &arguments ); ///< use natural args
    virtual RcPtr<Inst> subs                 ( const std::map<RcPtr<Inst>,RcPtr<Inst>> &map );
             
    virtual void        ct_rt_split          ( CompactReprWriter &cw, Vec<ExprData> &data_map ) const = 0;
    static RcPtr<Inst>  read_from            ( CompactReprReader &cr );
           
    void                find_natural_args_rec( Vec<Vec<std::pair<Str,RcPtr<Inst>>>> &nargs );
    void                clear_natural_args   ( );
    void                add_natural_arg      ( Str name, const RcPtr<Inst> &inst ); ///< name can be empty

    void                add_child            ( const RcPtr<Inst> &inst );
    int                 compare              ( const Inst &that ) const;
          
    static PI           rt_data_num          ( Vec<ExprData> &data_map, const Inst *inst, const std::function<ExprData::Val *()> &make_rt_data );
     
    PI                  ref_count = 0;
    Vec<Entry>          natural_args;      ///< used to say what to subs when using operator[] or operator()
    Vec<RcPtr<Inst>>    children;
    Vec<Inst*>          parents;
};


}


