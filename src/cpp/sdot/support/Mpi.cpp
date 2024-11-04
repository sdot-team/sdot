#include <string.h>
#include "Mpi.h"

SI32 Mpi::sum( SI32 value ) { return reduction( value, [&]( auto a, auto b ) { return a + b; } ); }
SI64 Mpi::sum( SI64 value ) { return reduction( value, [&]( auto a, auto b ) { return a + b; } ); }
PI32 Mpi::sum( PI32 value ) { return reduction( value, [&]( auto a, auto b ) { return a + b; } ); }
PI64 Mpi::sum( PI64 value ) { return reduction( value, [&]( auto a, auto b ) { return a + b; } ); }
FP64 Mpi::sum( FP64 value ) { return reduction( value, [&]( auto a, auto b ) { return a + b; } ); }

//
class NoMpi : public Mpi {
public:
    virtual int           rank                   () const override { return 0; }
    virtual int           size                   () const override { return 1; }

    virtual void          send                   ( const SI8  *data, PI count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const PI8  *data, PI count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const SI32 *data, PI count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const PI32 *data, PI count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const SI64 *data, PI count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const PI64 *data, PI count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const FP64 *data, PI count, int destination, int tag = 0 ) override {}

    virtual void          recv                   ( SI8  *data, PI count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( PI8  *data, PI count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( SI32 *data, PI count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( PI32 *data, PI count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( SI64 *data, PI count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( PI64 *data, PI count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( FP64 *data, PI count, int source, int tag = 0 ) override {}

    virtual void          gather                 ( SI32 *dst, const SI32 *src, PI count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }
    virtual void          gather                 ( SI64 *dst, const SI64 *src, PI count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }
    virtual void          gather                 ( PI32 *dst, const PI32 *src, PI count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }
    virtual void          gather                 ( PI64 *dst, const PI64 *src, PI count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }
    virtual void          gather                 ( FP64 *dst, const FP64 *src, PI count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }

    virtual void          all_gather             ( std::vector<std::vector<char>> &dst, const char *src, PI count ) override { dst.resize( 1 ); }
    virtual void          all_gather             ( std::vector<std::vector<int >> &dst, const int  *src, PI count ) override { dst.resize( 1 ); }

    virtual void          bcast                  ( SI8  *vec, PI count, int root = 0 ) override {}
    virtual void          bcast                  ( PI8  *vec, PI count, int root = 0 ) override {}
    virtual void          bcast                  ( SI32 *vec, PI count, int root = 0 ) override {}
    virtual void          bcast                  ( SI64 *vec, PI count, int root = 0 ) override {}
    virtual void          bcast                  ( PI32 *vec, PI count, int root = 0 ) override {}
    virtual void          bcast                  ( PI64 *vec, PI count, int root = 0 ) override {}
    virtual void          bcast                  ( FP64 *vec, PI count, int root = 0 ) override {}

    virtual PI            probe_size             ( int source, int tag = 0 ) override { return 0; }

    virtual void          selective_send_and_recv( std::vector<std::vector<char>> &ext, const std::vector<std::vector<int>> &needs, std::vector<char> &to_send ) override {}

    virtual void          cross_sends            ( std::vector<std::vector<std::uint8_t >> &dst, const std::vector<std::vector<std::uint8_t >> &src ) override { dst[ 0 ] = src[ 0 ]; }
    virtual void          cross_sends            ( std::vector<std::vector<PI32>> &dst, const std::vector<std::vector<PI32>> &src ) override { dst[ 0 ] = src[ 0 ]; }
    virtual void          cross_sends            ( std::vector<std::vector<PI64>> &dst, const std::vector<std::vector<PI64>> &src ) override { dst[ 0 ] = src[ 0 ]; }

    virtual void          barrier                () override {}

    virtual SI32          reduction              ( SI32 value, const std::function<SI32(SI32,SI32)> &f ) override { return value; }
    virtual SI64          reduction              ( SI64 value, const std::function<SI64(SI64,SI64)> &f ) override { return value; }
    virtual PI32          reduction              ( PI32 value, const std::function<PI32(PI32,PI32)> &f ) override { return value; }
    virtual PI64          reduction              ( PI64 value, const std::function<PI64(PI64,PI64)> &f ) override { return value; }
    virtual FP64          reduction              ( FP64 value, const std::function<FP64(FP64,FP64)> &f ) override { return value; }

    //    virtual void    partition  ( std::vector<int> &partition, const std::vector<PI> &node_off, const std::vector<PI> &edge_indices, const std::vector<PI> &edge_values, const std::vector<int> &edge_costs, const std::vector<double> &xyz, int dim, bool full_redistribution ) override { partition.resize( edge_indices.size() - 1 ); for( int &p : partition ) p = 0; }
    //    virtual void    partition  ( std::vector<int> &partition, const std::vector<PI> &node_off, const std::vector<double> &xyz, int dim ) override { partition.resize( xyz.size() / dim ); }
};

static NoMpi inst_mpi_abstraction;
Mpi *mpi = &inst_mpi_abstraction;


