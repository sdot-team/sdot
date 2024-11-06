#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/TODO.h>
#include "../support/Mpi.h"

namespace sdot {

/**
 * @brief MPI compatible reader to produce vectors
 * 
 *  `get_content` will give a portion of the data for each machine
 */
template<class T_,class Storage = Vec<T_>>
class KnownVecReader {
public:
    using   T                = T_;
      
    /**/    KnownVecReader   ( Storage &&data ) : data( std::move( data ) ) {}
   
    void    get_local_content( auto &&func ) const { if ( mpi->size() != 1 ) TODO; func( data ); }
    PI      global_size      () const { return data.size(); }

    void    display          ( auto &ds ) const { ds << data; }

    Storage data;
};

} // namespace sdot
