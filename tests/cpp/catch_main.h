#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../../src/cpp/sdot/support/to_string.h" // IWYU pragma: export

#define CHECK_REPR( A, B ) \
    CHECK( to_string( A ) == to_string( B ) )

//using namespace Vfs;

//#define CHECK_REPR_F( A, B, F ) \
//    CHECK( to_string( A, { .filter = F, .rec = 1 } ) == to_string( B, { .filter = F, .rec = 1 } ) )
