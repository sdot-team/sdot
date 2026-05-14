#pragma once

namespace sdot {

template<class TI,TI i>
struct CtdInt {
    static constexpr TI value = i;

    constexpr operator TI() const { return i; }
};

consteval auto ctd_add( int i, int j ) { if ( i < 0 ) return i; if ( j < 0 ) return j; return i + j;  }
consteval auto ctd_sub( int i, int j ) { if ( i < 0 ) return i; if ( j < 0 ) return j; return i - j;  }

// template<int i, int j> constexpr auto operator+( CtdInt<i>, CtdInt<j> ) { return CtdInt<ctd_add( i, j )>{}; }
// template<int i, int j> constexpr auto operator-( CtdInt<i>, CtdInt<j> ) { return CtdInt<ctd_sub( i, j )>{}; }

} // namespace sdot

/// 5_c will produce a CtdInt<5>()
// template<char... Digits>
// consteval auto operator""_c() {
//     constexpr int v = [] {
//         char ds[] = { Digits... };
//         int r = 0;
//         for ( char c : ds )
//             r = r * 10 + ( c - '0' );
//         return r;
//     }();
//     return sdot::CtdInt<v>{};
// }
