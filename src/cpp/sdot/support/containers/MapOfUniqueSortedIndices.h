#pragma once

#include "TensorView.h"

namespace sdot {

///
template<class TI>
struct IntWithOffset {
    HD void operator=  ( TI v ) { value = offset + v; }

    HD bool has_a_value() const { return value >= offset; }
    HD      operator TI() const { return value - offset; }
    HD bool operator!  () const { return value < offset; }

    TI      offset;
    TI&     value;
};

///
template<int ct_dim,class TV>
struct MapOfUniqueSortedIndices;

// Od
template<class TV>
struct MapOfUniqueSortedIndices<0,TV> {
    using TI = TV::value_type;

    HD MapOfUniqueSortedIndices( const auto &/* map_items */, const auto &/* nb_map_items */, TI /* max_inp_value */ ) {
    }

    HD void reserve_full_capacity() {
        offset = 1;
        value = 0;
    }

    HD void reserve( TI /* reservation */ ) {
        offset = 1;
        value = 0;
    }

    HD IntWithOffset<TI> operator[]( const Vector<TI,0> &/* key */ ) {
        return { offset, value };
    }

    HD void for_each_item( auto &&func ) const {
        if ( value >= offset )
            func( Vector<TI,0>( Values() ), value - offset );
    }

    TI offset; ///<
    TI value; ///<
};

// 1d
template<class TV>
struct MapOfUniqueSortedIndices<1,TV> {
    using TI = TV::value_type;

    HD MapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, TI max_inp_value ) : max_inp_value( max_inp_value ), values( map_items.offset( nb_map_items.post_increment( max_inp_value ) ) ) {
        // values.fill_with( 0 );
        next_offset = 1;
        offset = 0;
    }

    HD void reserve_full_capacity() {
        next_offset = 1;
        offset = 1;
        for( TI i = 0; i < max_inp_value; ++i )
            values[ i ] = 0;
    }

    HD void reserve( TI reservation ) {
        if ( next_offset == 1 )
            for( TI i = 0; i < max_inp_value; ++i )
                values[ i ] = 0;

        offset = next_offset;
        next_offset += reservation;
    }

    HD IntWithOffset<TI> operator[]( const Vector<TI,1> &key ) {
        return { offset, values[ key[ 0 ] ].ref() };
    }

    HD void for_each_item( auto &&func ) const {
        for( PI i = 0; i < values.size(); ++i )
            if ( values[ i ] >= offset )
                func( Vector<TI,1>( Values(), i ), values[ i ] - offset );
    }

    TI max_inp_value; ///<
    TI next_offset; ///<
    TI offset; ///<
    TV values; ///<
};

// 2d. TODO: hash table ?
template<class TV>
struct MapOfUniqueSortedIndices<2,TV> {
    using TI = TV::value_type;

    HD MapOfUniqueSortedIndices( const TV &map_items, auto &nb_map_items, TI max_inp_value ) : max_inp_value( max_inp_value ), values( map_items.offset( nb_map_items.post_increment( max_inp_value * max_inp_value ) ) ) {
        // values.fill_with( 0 );
        next_offset = 1;
        offset = 0;
    }

    HD void reserve_full_capacity() {
        next_offset = 1;
        offset = 1;
        for( TI i = 0; i < max_inp_value * max_inp_value; ++i )
            values[ i ] = 0;
    }

    HD void reserve( TI reservation ) {
        if ( next_offset == 1 )
            for( TI i = 0; i < max_inp_value * max_inp_value; ++i )
                values[ i ] = 0;

        offset = next_offset;
        next_offset += reservation;
    }

    HD IntWithOffset<TI> operator[]( const Vector<TI,2> &key ) {
        return { offset, values[ key[ 0 ] * max_inp_value + key[ 1 ] ].ref() };
    }

    HD void for_each_item( auto &&func ) const {
        for( PI i = 0; i < values.size(); ++i ) {
            for( PI j = 0; j < values.size(); ++j ) {
                PI k = i * max_inp_value + j;
                if ( values[ k ] >= offset )
                    func( Vector<TI,2>( Values(), i, j ), values[ k ] - offset );
            }
        }
    }

    TI max_inp_value; ///<
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

//     IntWithOffset operator[] ( const Vector<InputInt,-1,Arch> &key ) {
//         auto iter = values.find( key );
//         if ( iter == values.end() )
//             iter = values.insert( iter, { key, 0 } );
//         return { 1, iter->second };
//     }

// private:
//     using Map        = std::map<Vector<InputInt,-1,Arch>,PI64>;

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
