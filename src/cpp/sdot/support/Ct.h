#pragma once

 #include "common_macros.h"

namespace sdot {

template<class T,T i>
struct Ct {
    static constexpr T value = i;

    /**/ constexpr Ct( T value ) {
        ASSERT( value == i );
    }

    /**/ constexpr Ct() {
    }

    constexpr operator T() const {
        return i;
    }

    void display( auto &os ) const {
        os << "Ct(" << i  << ")";
    }
};

// Ct Ct
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator+( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 + v1 ); return Ct<TR,v0 + v1>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator-( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 - v1 ); return Ct<TR,v0 - v1>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator*( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 * v1 ); return Ct<TR,v0 * v1>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator/( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 / v1 ); return Ct<TR,v0 / v1>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator%( Ct<T0,v0>, Ct<T1,v1> ) { using TR = DECAYED_TYPE_OF( v0 % v1 ); return Ct<TR,v0 % v1>(); }

template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator&&( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 && v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator||( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 || v1)>(); }

template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator==( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 == v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator!=( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 != v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator<=( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 <= v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator>=( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 >= v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator< ( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 <  v1)>(); }
template<class T0,T0 v0,class T1,T1 v1> constexpr auto operator> ( Ct<T0,v0>, Ct<T1,v1> ) { return Ct<bool,(v0 >  v1)>(); }

// Ct T1
template<class T0,T0 v0,class T1> constexpr auto operator+( Ct<T0,v0>, T1 v1 ) { return v0 + v1; }
template<class T0,T0 v0,class T1> constexpr auto operator-( Ct<T0,v0>, T1 v1 ) { return v0 - v1; }
template<class T0,T0 v0,class T1> constexpr auto operator*( Ct<T0,v0>, T1 v1 ) { return v0 * v1; }
template<class T0,T0 v0,class T1> constexpr auto operator/( Ct<T0,v0>, T1 v1 ) { return v0 / v1; }
template<class T0,T0 v0,class T1> constexpr auto operator%( Ct<T0,v0>, T1 v1 ) { return v0 % v1; }

template<class T0,T0 v0,class T1> constexpr auto operator&&( Ct<T0,v0>, T1 v1 ) { return v0 && v1; }
template<class T0,T0 v0,class T1> constexpr auto operator||( Ct<T0,v0>, T1 v1 ) { return v0 || v1; }

template<class T0,T0 v0,class T1> constexpr auto operator==( Ct<T0,v0>, T1 v1 ) { return v0 == v1; }
template<class T0,T0 v0,class T1> constexpr auto operator!=( Ct<T0,v0>, T1 v1 ) { return v0 != v1; }
template<class T0,T0 v0,class T1> constexpr auto operator<=( Ct<T0,v0>, T1 v1 ) { return v0 <= v1; }
template<class T0,T0 v0,class T1> constexpr auto operator>=( Ct<T0,v0>, T1 v1 ) { return v0 >= v1; }
template<class T0,T0 v0,class T1> constexpr auto operator< ( Ct<T0,v0>, T1 v1 ) { return v0 <  v1; }
template<class T0,T0 v0,class T1> constexpr auto operator> ( Ct<T0,v0>, T1 v1 ) { return v0 >  v1; }


// T0 Ct
template<class T0,class T1,T1 v1> constexpr auto operator+( T0 v0, Ct<T1,v1> ) { return v0 + v1; }
template<class T0,class T1,T1 v1> constexpr auto operator-( T0 v0, Ct<T1,v1> ) { return v0 - v1; }
template<class T0,class T1,T1 v1> constexpr auto operator*( T0 v0, Ct<T1,v1> ) { return v0 * v1; }
template<class T0,class T1,T1 v1> constexpr auto operator/( T0 v0, Ct<T1,v1> ) { return v0 / v1; }
template<class T0,class T1,T1 v1> constexpr auto operator%( T0 v0, Ct<T1,v1> ) { return v0 % v1; }

template<class T0,class T1,T1 v1> constexpr auto operator&&( T0 v0, Ct<T1,v1> ) { return v0 && v1; }
template<class T0,class T1,T1 v1> constexpr auto operator||( T0 v0, Ct<T1,v1> ) { return v0 || v1; }
template<class T0,class T1,T1 v1> constexpr auto operator==( T0 v0, Ct<T1,v1> ) { return v0 == v1; }
template<class T0,class T1,T1 v1> constexpr auto operator!=( T0 v0, Ct<T1,v1> ) { return v0 != v1; }
template<class T0,class T1,T1 v1> constexpr auto operator<=( T0 v0, Ct<T1,v1> ) { return v0 <= v1; }
template<class T0,class T1,T1 v1> constexpr auto operator>=( T0 v0, Ct<T1,v1> ) { return v0 >= v1; }
template<class T0,class T1,T1 v1> constexpr auto operator< ( T0 v0, Ct<T1,v1> ) { return v0 <  v1; }
template<class T0,class T1,T1 v1> constexpr auto operator> ( T0 v0, Ct<T1,v1> ) { return v0 >  v1; }

} // namespace sdot
