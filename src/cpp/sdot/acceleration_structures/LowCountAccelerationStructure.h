#pragma once

#include <tl/support/containers/Opt.h>
#include "AccelerationStructure.h"
#include "AffineTransformation.h"
#include "../poom/PoomVec.h"

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TCell>
class LowCountAccelerationStructure : public AccelerationStructure<TCell> {
public:
    using                Pt                           = TCell::Pt;
    using                TF                           = Pt::value_type;
    static constexpr int nb_dims                      = Pt::ct_size;
    
    using                Trans                        = AffineTransformation<TF,nb_dims>; ///< transformation matrix + translation
    struct               Paw                          { Pt position; TF weight; void display( Displayer &ds ) const { DS_OJBECT( position, weight ); } };
                     
    /**/                 LowCountAccelerationStructure( const PoomVec<Pt> &positions, const PoomVec<TF> &weights, const Vec<Trans> &transformations );
    virtual             ~LowCountAccelerationStructure();
              
    virtual PI           recommended_nb_threads       () const override;
    virtual int          for_each_cell                ( const TCell &base_cell, const std::function<void( TCell &cell, int num_thread )> &f, int max_nb_threads = 0 ) override; ///< return the first non null value of `f( cell, num_thread )`, or 0. If max_nb_threads = 0, we take recommended_nb_threads() threads
    virtual PI           nb_cells                     () const override; ///< 
    virtual void         display                      ( Displayer &ds ) const override;

private:
    void                 make_cuts_from               ( PI i0, TCell &cell, Vec<PI> &buf );

    Vec<Paw>             position_and_weights;        ///<
    Vec<Trans>           transformations;             ///<
};         
         
} // namespace sdot

#include "LowCountAccelerationStructure.cxx"  // IWYU pragma: export
