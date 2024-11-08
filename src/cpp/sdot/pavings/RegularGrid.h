#pragma once

#include <tl/support/containers/Opt.h>
#include "AffineTransformation.h"

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TCell>
class RegularGrid {
public:
    using                Pt                    = TCell::Pt;
    using                TF                    = Pt::value_type;
    static constexpr int nb_dims               = Pt::ct_size;
   
    using                Trans                 = AffineTransformation<TF,nb_dims>; ///< transformation matrix + translation
    using                Nd                    = Vec<PI,nb_dims>;
                     
    /**/                 RegularGrid           ( const auto &points, const Vec<Trans> &transformations, TF nb_diracs_per_box = 30 );
              
    PI                   recommended_nb_threads() const;
    int                  for_each_cell         ( const TCell &base_cell, auto &&f, int max_nb_threads = 0 ); ///< return the first non null value of `f( cell, num_thread )`, or 0. If max_nb_threads = 0, we take recommended_nb_threads() threads
    void                 display               ( Displayer &ds ) const;
                  
private:   
    void                 make_cuts_from        ( PI b0, PI n0, TCell &cell, Vec<PI> &buf );
    PI                   end_index             () const;
    PI                   index                 ( const Pt &pos, int dim ) const;
    PI                   index                 ( const Pt &pos ) const;
            
    PI                   nb_base_points;       ///<     
    PI                   nb_glob_points;       ///<     
    Nd                   nb_divs;              ///<
    Vec<Pt,2>            limits;               ///< min and max pos
    Pt                   steps;                ///<
              
    Vec<Trans>           transformations;      ///<
    Vec<PI>              offsets;              ///<
    Vec<Pt>              points;               ///<
    Vec<PI>              inds;                 ///<
};         
         
} // namespace sdot

#include "RegularGrid.cxx"  // IWYU pragma: export
