#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/containers/Opt.h>
#include <tl/support/Displayer.h>

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TF,int nb_dims>
class RegularGrid {
public:
    using       Trans              = Vec<Vec<TF,nb_dims>,nb_dims+1>; ///< transformation matrix + translation
    using       Nd                 = Vec<PI,nb_dims>;
    using       Pt                 = Vec<TF,nb_dims>;
         
    /**/        RegularGrid        ( const auto &points, const Vec<Trans> &transformations, TF nb_diracs_per_box = 30 );
     
    PI          max_nb_threads     () const;
    int         for_each_cell      ( const auto &base_cell, const auto &weights, auto &&f ); ///< return the first non null value of `f( cell, num_thread )`, or 0
    void        display            ( Displayer &ds ) const;
      
private:  
    PI          end_index          () const;
    PI          index              ( const Pt &pos, int dim ) const;
    PI          index              ( const Pt &pos ) const;

    PI          nb_base_points;    ///<     
    PI          nb_glob_points;    ///<     
    Nd          nb_divs;           ///<
    Vec<Pt,2>   limits;            ///<
    Pt          steps;             ///<
  
    Vec<Trans>  transformations;   ///<
    Vec<PI>     offsets;           ///<
    Vec<Pt>     points;            ///<
    Vec<PI>     inds;              ///<
};

} // namespace sdot

#include "RegularGrid.cxx"  // IWYU pragma: export
