#pragma once

#include "PoomVecInst.h"

namespace sdot {
template<class T> class PoomVecInst;

/**
 * @brief Potentially Out of Memory vector
 * 
 * It can be used for instance to read data from a file, to handle data on several machines (MPI, ...) or to store data on harddisks if the RAM is not large enough.
 * 
 *  `get_modifiable_content` will give a portion of the data for each machine
 * 
 *  `get_content` will give a portion of the data for each machine
 * 
 * Internally, it is a ref-counted pointer to a PoomVecInst<T>
 */
template<class T_>
class PoomVec {
public:
    using       T                    = T_;
    using       Inst                 = PoomVecInst<T>;

    /* */       PoomVec              ( CstSpan<T> data );

    void        get_values_by_chuncks( const std::function<void( CstSpanView<T> )> &func, PI beg, PI end ); ///< 
    void        get_values_by_chuncks( const std::function<void( CstSpanView<T> )> &func ); ///< get chuncks for all the data
     
    PI          size                 () const;
    
    void        display              ( Displayer &ds ) const { ds << inst; }

private:
    RcPtr<Inst> inst;
};

} // namespace sdot

#include "PoomVec.cxx" // IWYU pragma: export
