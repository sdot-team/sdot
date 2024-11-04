#pragma once

// #include "MyPrint.h"
#include <tl/support/common_types.h>
#include <functional>
// #include <vector>

/**
  Allows to use mpi semantic even if mpi is not loaded.

  Use #include "Mpi.h" and mpi.init( argc, argv ) to use MPI
*/
class Mpi {
public:
    bool                  master                 () const { return rank() == 0; }
    virtual int           rank                   () const = 0;
    virtual int           size                   () const = 0;

    virtual void          send                   ( const SI8  *data, PI count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const PI8  *data, PI count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const SI32 *data, PI count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const PI32 *data, PI count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const SI64 *data, PI count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const PI64 *data, PI count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const FP64 *data, PI count, int destination, int tag = 0 ) = 0;

    virtual void          recv                   ( SI8  *data, PI count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( PI8  *data, PI count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( SI32 *data, PI count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( PI32 *data, PI count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( SI64 *data, PI count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( PI64 *data, PI count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( FP64 *data, PI count, int source, int tag = 0 ) = 0;

    virtual void          gather                 ( SI32 *dst, const SI32 *src, PI count, int root = 0 ) = 0;
    virtual void          gather                 ( SI64 *dst, const SI64 *src, PI count, int root = 0 ) = 0;
    virtual void          gather                 ( PI32 *dst, const PI32 *src, PI count, int root = 0 ) = 0;
    virtual void          gather                 ( PI64 *dst, const PI64 *src, PI count, int root = 0 ) = 0;
    virtual void          gather                 ( FP64 *dst, const FP64 *src, PI count, int root = 0 ) = 0;

    virtual void          all_gather             ( std::vector<std::vector<char>> &dst, const char *src, PI count ) = 0;
    virtual void          all_gather             ( std::vector<std::vector<int >> &dst, const int  *src, PI count ) = 0;

    virtual void          bcast                  ( SI8  *vec, PI count, int root = 0 ) = 0;
    virtual void          bcast                  ( PI8  *vec, PI count, int root = 0 ) = 0;
    virtual void          bcast                  ( SI32 *vec, PI count, int root = 0 ) = 0;
    virtual void          bcast                  ( SI64 *vec, PI count, int root = 0 ) = 0;
    virtual void          bcast                  ( PI32 *vec, PI count, int root = 0 ) = 0;
    virtual void          bcast                  ( PI64 *vec, PI count, int root = 0 ) = 0;
    virtual void          bcast                  ( FP64 *vec, PI count, int root = 0 ) = 0;

    virtual void          barrier                () = 0;

    virtual void          selective_send_and_recv( std::vector<std::vector<char>> &ext, const std::vector<std::vector<int>> &needs, std::vector<char> &to_send ) = 0;

    virtual void          cross_sends            ( std::vector<std::vector<std::uint8_t >> &dst, const std::vector<std::vector<std::uint8_t >> &src ) = 0;
    virtual void          cross_sends            ( std::vector<std::vector<PI32>> &dst, const std::vector<std::vector<PI32>> &src ) = 0;
    virtual void          cross_sends            ( std::vector<std::vector<PI64>> &dst, const std::vector<std::vector<PI64>> &src ) = 0;

    virtual PI            probe_size             ( int source, int tag = 0 ) = 0;
                     
    virtual SI32          reduction              ( SI32 value, const std::function<SI32(SI32,SI32)> &f ) = 0;
    virtual SI64          reduction              ( SI64 value, const std::function<SI64(SI64,SI64)> &f ) = 0;
    virtual PI32          reduction              ( PI32 value, const std::function<PI32(PI32,PI32)> &f ) = 0;
    virtual PI64          reduction              ( PI64 value, const std::function<PI64(PI64,PI64)> &f ) = 0;
    virtual FP64          reduction              ( FP64 value, const std::function<FP64(FP64,FP64)> &f ) = 0;
         
    virtual SI32          sum                    ( SI32 value );
    virtual SI64          sum                    ( SI64 value );
    virtual PI32          sum                    ( PI32 value );
    virtual PI64          sum                    ( PI64 value );
    virtual FP64          sum                    ( FP64 value );
         
    // virtual void       partition  ( std::vector<int> &partition, const std::vector<PI> &node_offsets, const std::vector<PI> &edge_indices, const std::vector<PI> &edge_values, const std::vector<int> &edge_costs, const std::vector<FP64> &xyz, int dim, bool full_redistribution ) = 0;
    // virtual void       partition  ( std::vector<int> &partition, const std::vector<PI> &node_offsets, const std::vector<FP64> &xyz, int dim ) = 0;
};

extern Mpi *mpi;

// template<class OS,class... Args> void ___my_mpi_print( OS &os, const char *str, const Args &...args ) {
//     // get local msg
//     std::ostringstream ss;
//     __my_print( ss, str, ", ", args... );
//     std::string msg = ss.str();

//     // send data (if rank != 0) or display it
//     if ( mpi->rank() ) {
//         mpi->send( (const std::int8_t *)msg.data(), msg.size(), 0, mpi->rank() );
//     } else {
//         // message for rank 0
//         if ( mpi->size() )
//             os << "rank 0: ";
//         os << msg;

//         // other messages
//         for( int rank = 1; rank < mpi->size(); ++rank ) {
//             std::string str( mpi->probe_size( rank, rank ), ' ' );
//             mpi->recv( (std::int8_t *)str.data(), str.size(), rank, rank );
//             os << "rank " << rank << ": " << str;
//         }
//     }

//     // we don't want other displays to interfere
//     mpi->barrier();
// }

// template<class OS,class... Args> void ___my_mpi_print_0( OS &os, const char *str, const Args &...args ) {
//     // get local msg
//     std::ostringstream ss;
//     __my_print( ss, str, ", ", args... );

//     if ( mpi->rank() == 0 )
//         os << ss.str();
// }

// #define PNMPI( ... ) ___my_mpi_print( std::cout, #__VA_ARGS__ " ->\n" , __VA_ARGS__ )
// #define PMPI( ... ) ___my_mpi_print( std::cout, #__VA_ARGS__ " -> " , __VA_ARGS__ )
// #define PMPI_0( ... ) ___my_mpi_print_0( std::cout, #__VA_ARGS__ " -> " , __VA_ARGS__ )
