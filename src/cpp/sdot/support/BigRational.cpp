#include "BigRational.h"
#include <boost/multiprecision/detail/default_ops.hpp>
#include <boost/multiprecision/detail/integer_ops.hpp>

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
    ds.start_object();
    ds.append_attribute( "num", num );
    ds.append_attribute( "den", den );
    ds.append_attribute( "exp", exp );
    ds.end_object();
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

BigRational abs( const BigRational &a ) {
    return { BigRational::FromNormalizedData(), boost::multiprecision::abs( a.num ), a.den, a.exp };
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

} // namespace sdot
