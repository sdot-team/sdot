// #include <boost/multiprecision/detail/default_ops.hpp>
// #include <boost/multiprecision/detail/integer_ops.hpp>
#include <tl/support/string/CompactReprReader.h>
#include <tl/support/string/CompactReprWriter.h>
#include <tl/support/ASSERT.h>
#include "BigRational.h"
#include "asimd/support/N.h"

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

    bool is_neg = false;
    if ( str.starts_with( '-' ) ) {
        str.remove_prefix( 1 );
        is_neg = true;
    }

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

    if ( is_neg )
        num = - num;

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
        res += "p" + std::to_string( exp );
    ds << res;
}

BigRational::operator bool() const {
    return bool( num );
}

BigRational::BI ceil( const BigRational &a ) {
    BigRational::BI n = a.num;
    BigRational::BI d = a.den;
    if ( a.exp > 0 )
        n <<= a.exp;
    else
        d <<= - a.exp;
    if ( n >= 0 )
        return n / d;
    return ( n - d + 1 ) / d;
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
    if ( b.is_integer() ) {
        if ( b > 0 ) {
            BigRational res = a; // OPTIMIZE
            for( PI i = 0, n = PI( ceil( b ) ); i < n; ++i )
                res *= a;
            return res;
        }

        if ( b < 0 ) {
            BigRational res = 1 / a; // OPTIMIZE
            for( PI i = 0, n = PI( - ceil( b ) ); i < n; ++i )
                res *= a;
            return res;
        }

        return 1;
    }

    TODO;
    return 0;
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
    return 0;
}

BigRational BigRationalFrom<FP64>::create( FP64 value ) {
    auto res = ( boost::multiprecision::cpp_int( 1 ) << 52 ) + ( reinterpret_cast<const PI64 &>( value ) & ( ( PI64( 1 ) << 52 ) - 1 ) ); 
    SI64 bex = ( reinterpret_cast<const PI64 &>( value ) >> 52 ) & ( ( PI64( 1 ) << 11 ) - 1 );

    // zero ?
    if ( bex == 0 )
        return { BigRational::FromNormalizedData(), 0, 1, 0 };

    bool sgn = reinterpret_cast<const PI64 &>( value ) & ( PI64( 1 ) << 63 );
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

BigRational BigRational::read_from( CompactReprReader &cr ) {
    const bool is_int( cr.read_positive_int( 2 ) );
    if ( is_int ) {
        const bool num_is_pos( cr.read_positive_int( 2 ) );
        return num_is_pos ? cr.read_positive_int() : - cr.read_positive_int();
    } else {
        const bool num_is_pos( cr.read_positive_int( 2 ) );
        const bool exp_is_pos( cr.read_positive_int( 2 ) );
        const BI num( num_is_pos ? cr.read_positive_int() : - cr.read_positive_int() );
        const BI den( cr.read_positive_int() );
        const TE exp( exp_is_pos ? cr.read_positive_int() : - cr.read_positive_int() );
        return { num, den, exp };
    }
}

void BigRational::write_to( CompactReprWriter &cw ) const {
    cw.write_positive_int( is_integer(), 2 );

    if ( is_integer() ) {
        cw.write_positive_int( is_positive_or_null(), 2 );
        cw.write_positive_int( boost::multiprecision::abs( num ) << exp );
    } else {
        cw.write_positive_int( is_positive_or_null(), 2 );
        cw.write_positive_int( exp >= 0, 2 );
        cw.write_positive_int( boost::multiprecision::abs( num ) );
        cw.write_positive_int( den );
        cw.write_positive_int( std::abs( exp ) );

    }
}

} // namespace sdot
