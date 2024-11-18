// #include <boost/multiprecision/detail/default_ops.hpp>
// #include <boost/multiprecision/detail/integer_ops.hpp>
#include <string>
#include <tl/support/string/read_number.h>
#include <tl/support/ASSERT.h>
#include "BigRational.h"

namespace sdot {

void BigRational::normalize_div() {
    using boost::multiprecision::gcd;
    using boost::multiprecision::abs;

    // sign
    if ( den < 0 ) {
        num = - num;
        den = - den;
    }

    BI g = boost::multiprecision::gcd( abs( num ), den );
    num /= g;
    den /= g;
}

void BigRational::normalize_exp() {
    while ( num && ! boost::multiprecision::bit_test( num, 0 ) ) {
        num >>= 1;
        ++exp;
    }

    while ( den && ! boost::multiprecision::bit_test( den, 0 ) ) {
        den >>= 1;
        --exp;
    }
}

void BigRational::normalize_all() {
    normalize_div();
    normalize_exp();
}


BigRational::BigRational( FromNormalizedData, BI num, BI den, SI64 exp ) : num( num ), den( den ), exp( exp ) {
}

BigRational::BigRational( FromNormalizedDiv, BI num, BI den, SI64 exp ) : num( num ), den( den ), exp( exp ) {
    normalize_exp();
}

BigRational::BigRational( BI num, BI den, SI64 exp ) : num( num ), den( den ), exp( exp ) {
    normalize_all();
}

BigRational::BigRational( BI num, BI den ) : num( num ), den( den ), exp( 0 ) {
    normalize_all();
}

BigRational::BigRational( BI num ) : num( num ), den( 1 ), exp( 0 ) {
    normalize_exp();
}

BigRational::BigRational( StrView str ) : num( 0 ), den( 1 ), exp( 0 ) {
    auto read_den = [&]( PI &i ) {
        den = 0;
        while( ++i < str.size() ) {
            char c = str[ i ];

            if ( c >= '0' && c <= '9' ) {
                den = 10 * den + ( c - '0' );
                continue;
            }

            if ( c == 'e' ) {
                TODO;
                break;
            }

            TODO;
        }
    };

    for( PI i = 0; i < str.size(); ) {
        char c = str[ i ];

        if ( c >= '0' && c <= '9' ) {
            *this = 10 * *this + ( c - '0' );
            ++i;
            continue;
        }

        if ( c == '/' ) {
            read_den( i );
            break;            
        }

        TODO;
    }

    normalize_all();
}

BigRational::BigRational( const char *str ) : BigRational( StrView( str ) ) {
}

BigRational::BigRational() : num( 0 ), den( 1 ), exp( 0 ) {
}

BigRational::BigRational( const BigRational &that ) : num( that.num ), den( that.den ), exp( that.exp ) {
}

BigRational::BigRational( BigRational &&that ) : num( std::move( that.num ) ), den( std::move( that.den ) ), exp( std::move( that.exp ) ) {
}

BigRational &BigRational::operator=( const BigRational &that ) {
    num = that.num;
    den = that.den;
    exp = that.exp;
    return *this;
}

BigRational &BigRational::operator=( BigRational &&that ) {
    num = std::move( that.num );
    den = std::move( that.den );
    exp = std::move( that.exp );
    return *this;
}

void BigRational::display( Displayer &ds ) const {
    Str res = num.str();
    if ( den != 1 )
        res += "/" + den.str();
    if ( exp != 0 )
        res += "e" + std::to_string( exp );
    ds << res;
}

BigRational::operator bool() const {
    return bool( num );
}

BigRational operator+( const BigRational &a, const BigRational &b ) {
    auto miexp = std::min( a.exp, b.exp );
    auto a_num = a.num << ( a.exp - miexp );
    auto b_num = b.num << ( b.exp - miexp );
    return { a_num * b.den + b_num * a.den, a.den * b.den, miexp };
}

BigRational operator-( const BigRational &a, const BigRational &b ) {
    auto m_exp = std::min( a.exp, b.exp );
    auto a_num = a.num << ( a.exp - m_exp );
    auto b_num = b.num << ( b.exp - m_exp );
    return { a_num * b.den - b_num * a.den, a.den * b.den, m_exp };
}

BigRational operator*( const BigRational &a, const BigRational &b ) {
    return { a.num * b.num, a.den * b.den, a.exp + b.exp };
}

BigRational operator/( const BigRational &a, const BigRational &b ) {
    return { a.num * b.den, a.den * b.num, a.exp - b.exp };
}

bool operator==( const BigRational &a, const BigRational &b ) {
    return std::tie( a.num, a.den, a.exp ) == std::tie( b.num, b.den, b.exp );
}

bool operator!=( const BigRational &a, const BigRational &b ) {
    return std::tie( a.num, a.den, a.exp ) != std::tie( b.num, b.den, b.exp );
}

bool operator<=( const BigRational &a, const BigRational &b ) {
    auto m_exp = std::min( a.exp, b.exp );
    auto a_num = a.num << ( a.exp - m_exp );
    auto b_num = b.num << ( b.exp - m_exp );
    return a_num * b.den <= b_num * a.den;
}

bool operator>=( const BigRational &a, const BigRational &b ) {
    auto m_exp = std::min( a.exp, b.exp );
    auto a_num = a.num << ( a.exp - m_exp );
    auto b_num = b.num << ( b.exp - m_exp );
    return a_num * b.den >= b_num * a.den;
}

bool operator<( const BigRational &a, const BigRational &b ) {
    auto m_exp = std::min( a.exp, b.exp );
    auto a_num = a.num << ( a.exp - m_exp );
    auto b_num = b.num << ( b.exp - m_exp );
    return a_num * b.den < b_num * a.den;
}

bool operator>( const BigRational &a, const BigRational &b ) {
    auto m_exp = std::min( a.exp, b.exp );
    auto a_num = a.num << ( a.exp - m_exp );
    auto b_num = b.num << ( b.exp - m_exp );
    return a_num * b.den > b_num * a.den;
}

BigRational &BigRational::operator+=( const BigRational &that ) {
    *this = *this + that;
    
    return *this;
}

BigRational &BigRational::operator-=( const BigRational &that ) {
    *this = *this - that;
    
    return *this;
}

BigRational &BigRational::operator*=( const BigRational &that ) {
    num *= that.num;
    den *= that.den;
    exp += that.exp;
    normalize_all();
    
    return *this;
}

BigRational &BigRational::operator/=( const BigRational &that ) {
    num *= that.den;
    den *= that.num;
    exp -= that.exp;
    normalize_all();
    
    return *this;
}

BigRational BigRational::operator-() const {
    return { FromNormalizedData(), - num, den, exp };
}

BigRational pow( const BigRational &a, const BigRational &b ) {
    // if ( )
    TODO;
}

BigRational abs( const BigRational &a ) {
    return { BigRational::FromNormalizedData(), boost::multiprecision::abs( a.num ), a.den, a.exp };
}

BigRational BigRational::pow( const BigRational &that ) const {
    if ( that == 0 )
        return 1;
    if ( that == 1 )
        return *this;
    TODO;
}

BigRational BigRationalFrom<FP64>::create( FP64 value ) {
    auto res = ( boost::multiprecision::cpp_int( 1 ) << 52 ) + ( reinterpret_cast<const PI64 &>( value ) & ( ( 1ul << 52 ) - 1 ) ); 
    SI64 bex = ( reinterpret_cast<const PI64 &>( value ) >> 52 ) & ( ( 1ul << 11 ) - 1 );

    // zero ?
    if ( bex == 0 )
        return { BigRational::FromNormalizedData(), 0, 1, 0 };

    bool sgn = reinterpret_cast<const PI64 &>( value ) & ( 1ul << 63 );
    return { BigRational::FromNormalizedDiv(), sgn ? - res : res, 1, bex - ( 1023 + 52 ) };
}

int BigRational::compare( const sdot::BigRational &a, const sdot::BigRational &b ) {
    auto m_exp = std::min( a.exp, b.exp );
    auto a_num = a.num << ( a.exp - m_exp );
    auto b_num = b.num << ( b.exp - m_exp );
    auto m0 = a_num * b.den;
    auto m1 = b_num * a.den;
    if ( m0 < m1 )
        return -1;
    if ( m0 > m1 )
        return +1;
    return 0;   
}

Str BigRational::compact_str_repr( PI p, PI m ) const {
    // subtypes: full_rational , int
    p += m * is_integer(); m *= 2;
    // pos or neg num
    p += m * positive_or_null(); m *= 2;

    //
    if ( is_integer() ) {
        Str num_str = BI( boost::multiprecision::abs( num ) << exp ).str();
        p += m * num_str.size();
  
        return std::to_string( p ) + "_" + num_str;
    } else {
        // exp is positive 
        p += m * ( exp >= 0 ); m *= 2;

        Str num_str = BI( boost::multiprecision::abs( num ) ).str();
        Str den_str = den.str();
        p += m * num_str.size();
  
        return std::to_string( p ) + "_" + num_str +
               std::to_string( den_str.size() ) + "_" + den_str +
               std::to_string( std::abs( exp ) );
    }
}

BigRational BigRational::from_compact_str( StrView &res, PI m ) {
    PI p = read_number<PI>( res ); ASSERT( res.starts_with( '_' ) ); res.remove_prefix( 1 );
    PI sub_type = p % 2; p /= 2;

    // int
    if ( sub_type == 1 ) {
        bool positive_num = p % 2; p /= 2;
        
        BI num( res.substr( 0, p ) ); res.remove_prefix( p );

        return positive_num ? num : - num;
    } else {
        bool positive_num = p % 2; p /= 2;
        bool positive_exp = p % 2; p /= 2;

        BI num( res.substr( 0, p ) ); res.remove_prefix( p );

        PI sden = read_number<PI>( res ); ASSERT( res.starts_with( '_' ) ); res.remove_prefix( 1 );
        BI den( res.substr( 0, sden ) ); res.remove_prefix( sden );
        TE exp = read_number<TE>( res );

        return { positive_num ? num : -num, den, positive_exp ? exp : - exp };
    }
}

} // namespace sdot
