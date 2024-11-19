#pragma once

#include <tl/support/Displayer.h>
#include <functional>

namespace sdot {

/**
 * @brief 
 * 
 */
template<class TCell>
class AccelerationStructure {
public:
    virtual      ~AccelerationStructure () {}
              
    virtual PI    recommended_nb_threads() const = 0;
    virtual int   for_each_cell         ( const TCell &base_cell, const std::function<void( TCell &cell, int num_thread )> &f, int max_nb_threads = 0 ) = 0; ///< return the first non null value of `f( cell, num_thread )`, or 0. If max_nb_threads = 0, we take recommended_nb_threads() threads
    virtual PI    nb_cells              () const = 0; ///< 
    virtual void  display               ( Displayer &ds ) const = 0;
};         
         
} // namespace sdot
