#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>

// #include <vfs/support/string/to_string.h> // IWYU pragma: export
// #include <vfs/support/make_Vec.h> // IWYU pragma: export
#include <tl/support/string/to_string.h> // IWYU pragma: export
#include <tl/support/compare.h> // IWYU pragma: export
#include <tl/support/P.h> // IWYU pragma: export

#define CHECK_REPR( A, B ) \
    CHECK( to_string( A, { .always_display_delimiters = true } ) == to_string( B, { .always_display_delimiters = true } ) )

auto _sorted( const auto &v ) { auto r = v; std::sort( r.begin(), r.end(), Less() ); return r; }

#define CHECK_REPR_SORTED( A, B ) \
    CHECK( to_string( _sorted( A ) ) == to_string( _sorted( B ) ) )

//using namespace Vfs;

//#define CHECK_REPR_F( A, B, F ) \
//    CHECK( to_string( A, { .filter = F, .rec = 1 } ) == to_string( B, { .filter = F, .rec = 1 } ) )

