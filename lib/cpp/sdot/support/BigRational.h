#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <tl/support/Displayer.h>
#include <tl/support/compare.h>
#include <Eigen/Dense>
#include <cmath>
class CompactReprWriter;
class CompactReprReader;

namespace sdot {
T_T struct BigRationalFrom;

/**
 * @brief a (slow) rational class represented as num / den * 2 ^ exp
 * 
 */
class BigRational {
public: 
    struct             FromNormalizedData {}; ///< can be used by the ctor to say that num, den and exp are already normalized
    struct             FromNormalizedDiv  {}; ///< can be used by the ctor to say that num and den already normalized
    using              BI                 = boost::multiprecision::cpp_int; ///< num and den types
    using              TE                 = SI64; ///< exp type

    /**/               BigRational        ( FromNormalizedData, BI num = 0, BI den = 1, TE exp = 0 );
    /**/               BigRational        ( FromNormalizedDiv, BI num = 0, BI den = 1, TE exp = 0 );
    /**/               BigRational        ( BI num, BI den, TE exp );
    /**/               BigRational        ( BI num, BI den );
    /**/               BigRational        ( const char *str );
    /**/               BigRational        ( StrView str );
    /**/               BigRational        ( BI num );
    /**/               BigRational        ();
      
    /**/               BigRational        ( const BigRational &that );
    /**/               BigRational        ( BigRational &&that );
      
    /**/               BigRational        ( const auto &value ) requires requires { BigRationalFrom<DECAYED_TYPE_OF( value )>::create( value ); } { *this = BigRationalFrom<DECAYED_TYPE_OF( value )>::create( value ); }
              
    BigRational&       operator=          ( const BigRational &that );
    BigRational&       operator=          ( BigRational &&that );

    BI                 denominator        () const { return den; }
    BI                 numerator          () const { return num; }
    BI                 exponent           () const { return exp; }

    bool               is_positive_or_null() const { return num >= 0; }
    bool               is_integer         () const { return den == 1 && exp >= 0; }

    void               display            ( Displayer &ds ) const;
   
    explicit           operator bool      () const;
    T_T explicit       operator T         () const;
        
    friend bool        operator==         ( const BigRational &a, const BigRational &b );
    friend bool        operator!=         ( const BigRational &a, const BigRational &b );
    friend bool        operator<=         ( const BigRational &a, const BigRational &b );
    friend bool        operator>=         ( const BigRational &a, const BigRational &b );
    friend bool        operator<          ( const BigRational &a, const BigRational &b );
    friend bool        operator>          ( const BigRational &a, const BigRational &b );
           
    BigRational&       operator+=         ( const BigRational &that );
    BigRational&       operator-=         ( const BigRational &that );
    BigRational&       operator*=         ( const BigRational &that );
    BigRational&       operator/=         ( const BigRational &that );
        
    friend BigRational operator+          ( const BigRational &a, const BigRational &b );
    friend BigRational operator-          ( const BigRational &a, const BigRational &b );
    friend BigRational operator*          ( const BigRational &a, const BigRational &b );
    friend BigRational operator/          ( const BigRational &a, const BigRational &b );
        
    BigRational        operator-          () const;

    static int         compare            ( const BigRational &a, const BigRational &b );
    friend BI          ceil               ( const BigRational &a );
    friend BigRational pow                ( const BigRational &a, const BigRational &b );
    friend BigRational abs                ( const BigRational &a );

    int                compare            ( const BigRational &that ) const { return BigRational::compare( *this, that ); }
    BigRational        pow                ( const BigRational &that ) const;
        
    static BigRational read_from          ( CompactReprReader &cr );
    void               write_to           ( CompactReprWriter &cw ) const;

private:        
    void               normalize_all      ();
    void               normalize_div      ();
    void               normalize_exp      ();

    BI                 num;
    BI                 den;
    TE                 exp;
};

// template definitions
T_T BigRational::operator T() const { 
    return T( num << std::max( + exp, 0l ) ) / T( den << std::max( - exp, 0l ) );
}


// create a BigRational from an integer
template<class I> requires std::is_integral_v<I> struct BigRationalFrom<I> {
    static BigRational create( const I &value ) {
        return { BigRational::FromNormalizedDiv(), value, 1 };
    }
};

// create a BigRational from a floating point value
template<> struct BigRationalFrom<FP64> { static BigRational create( FP64 value ); };

} // namespace sdot

namespace Eigen {
    template<> struct NumTraits<sdot::BigRational> : GenericNumTraits<sdot::BigRational> {
        typedef sdot::BigRational NonInteger;
        typedef sdot::BigRational Nested;
        typedef sdot::BigRational Real;

        static inline Real dummy_precision() { return 0; }
        static inline Real digits10() { return 0; }
        static inline Real epsilon() { return 0; }

        enum {
            IsInteger = 0,
            IsSigned = 1,
            IsComplex = 0,
            RequireInitialization = 1,
            ReadCost = 6,
            AddCost = 150,
            MulCost = 100
        };
    };
}
