#pragma once

// #include <tl/support/containers/Vec.h>
// #include <tl/support/Displayer.h>
// #include <tl/support/compare.h>
// #include "VecForCapa.h"
#include "TensorView.h"
#include "DsVec.h"

namespace sdot {

///
template<class TI>
struct IntWithOffset {
    void operator=  ( TI v ) { value = offset + v; }

    bool has_a_value() const { return value >= offset; }
    bool operator!  () const { return value < offset; }
    operator TI     () const { return value - offset; }

    TI  offset;
    TI& value;
};

///
template<int ct_dim,class TI,class Arch>
struct MapOfUniqueSortedIndices;

// Od
template<class TI,class Arch>
struct MapOfUniqueSortedIndices<0,TI,Arch> {
    /**/ MapOfUniqueSortedIndices( const auto &/* map_items */, const auto &/* nb_map_items */, int /* dim */, TI /* max_inp_value */ ) {
    }

    void reserve_full_capacity() {
        offset = 1;
        value = 0;
    }

    void reserve( TI /* reservation */ ) {
        offset = 1;
        value = 0;
    }

    IntWithOffset<TI> operator[]( const DsVec<TI,0,Arch> &/* key */ ) {
        return { offset, value };
    }

    void for_each_item( auto &&func ) const {
        if ( value >= offset )
            func( DsVec<TI,0,Arch>( Values() ), value - offset );
    }

    TI offset; ///<
    TI value; ///<
};

// 1d
template<class TI,class Arch>
struct MapOfUniqueSortedIndices<1,TI,Arch> {
    using TV = TensorView<TI,1,Arch>;

    /**/ MapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, int /* dim */, TI max_inp_value ) : values( map_items ) {
        nb_map_items = max_inp_value;
        next_offset = 1;
        offset = 0;
    }

    void reserve_full_capacity() {
        next_offset = 1;
        offset = 1;
        for( auto &value : values )
            value = 0;
    }

    void reserve( TI reservation ) {
        if ( next_offset == 1 )
            for( auto &value : values )
                value = 0;

        offset = next_offset;
        next_offset += reservation;
    }

    IntWithOffset<TI> operator[]( const DsVec<TI,1,Arch> &key ) {
        return { offset, values[ key[ 0 ] ] };
    }

    void for_each_item( auto &&func ) const {
        for( PI i = 0; i < values.size(); ++i )
            if ( values[ i ] >= offset )
                func( DsVec<TI,1,Arch>( Values(), i ), values[ i ] - offset );
    }

    TI next_offset; ///<
    TI offset; ///<
    TV values; ///<
};


/**
 * map[ Vec<PII>(...) ] => ... where items in Vec<PII>(...) are strictly increasing
 *
 * This version is for the generic case (it uses std::map which may be slow).
*/
// template<class InputInt,int ct_dim,class Arch>
// class MapOfUniqueSortedIndices {
// public:
//     /**/  MapOfUniqueSortedIndices( PI /* dim */ ) {}

//     void  prepare_for             ( InputInt /*max_input_value*/, PI64 /* max_output_value */ ) { values.clear(); }

//     IntWithOffset operator[] ( const DsVec<InputInt,-1,Arch> &key ) {
//         auto iter = values.find( key );
//         if ( iter == values.end() )
//             iter = values.insert( iter, { key, 0 } );
//         return { 1, iter->second };
//     }

// private:
//     using Map        = std::map<DsVec<InputInt,-1,Arch>,PI64>;

//     Map   values;    ///<
// };

// /// s == 0
// template<class PII,class PIO>
// class MapOfUniqueSortedIndices<0,PII,PIO> {
// public:
//     void  prepare_for ( PII /*max_PII_value*/ ) {}
//     PIO&  operator[]  ( Vec<PII,0> ) { return value; }
//     void  display     ( Displayer &ds ) const { ds << value; }

// private:
//     PIO   value = 0;  ///<
// };

// /// s == 1
// template<class PII,class PIO>
// class MapOfUniqueSortedIndices<1,PII,PIO> {
// public:
//     void   prepare_for( PII max_PII_value ) { values.reserve( max_PII_value, 0 ); }
//     PIO&   operator[] ( Vec<PII,1> a ) { return values[ a[ 0 ] ]; }
//     void   display    ( Displayer &ds ) const { ds << values; }

// private:
//     using  Map        = VecForCapa<PIO>;

//     Map    values;    ///<
// };

// /// s == 2
// template<class PII,class PIO>
// struct MapOfUniqueSortedIndices<2,PII,PIO> {
//     void   prepare_for             ( PI max_PII_value ) { values.reserve( ( max_PII_value - 1 ) * max_PII_value / 2, 0 ); }
//     PIO&   operator[]              ( const Vec<PII,2> &a ) { return values[ ( a[ 1 ] - 1 ) * a[ 1 ] / 2 + a[ 0 ] ]; }
//     void   display                 ( Displayer &ds ) const { ds << values; }

//     PIO&   at_without_index        ( const Vec<PII,3> &a, auto index ) {
//          const auto a0 = a[ 0 + ( index <= 0 ) ], a1 = a[ 1 + ( index <= 1 ) ];
//          return values[ ( a1 - 1 ) * a1 / 2 + a0 ];
//     }

// private:
//     using  Map      = VecForCapa<PIO>;

//     Map    values;
// };

// /// s == 2
// template<class PII,class PIO>
// struct MapOfUniqueSortedIndices<2,PII,PIO> {
//     void   prepare_for( PI max_PII_value ) { size = max_PII_value; values.reserve( max_PII_value * max_PII_value, 0 ); }
//     PIO&   operator[] ( const Vec<PII,2> &a ) { return values[ a[ 1 ] * size + a[ 0 ] ]; }
//     void   display    ( Displayer &ds ) const { ds << values; }

// private:
//     using  Map        = VecForCapa<PIO>;

//     Map    values;
//     PI     size;
// };

} // namespace sdot
