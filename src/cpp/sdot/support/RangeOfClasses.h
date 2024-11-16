#pragma once

#include <tl/support/containers/CtInt.h>
#include <tl/support/Displayer.h>

/*
*/
template<template<int> typename CT,int beg,int end>
class RangeOfClasses {
public:
    using     Tail          = RangeOfClasses<CT,beg+1,end>;
    using     Head          = CT<beg>;

    /**/      RangeOfClasses( const auto &...args ) : head( args... ), tail( args... ) {}

    void      for_each_item ( auto &&f ) const { f( head ); tail.for_each_item( FORWARD( f ) ); }
    void      for_each_item ( auto &&f ) { f( head ); tail.for_each_item( FORWARD( f ) ); }
    T_i auto& operator[]    ( CtInt<i> ) { return tail[ CtInt<i>() ]; }
    auto&     operator[]    ( CtInt<beg> ) { return head; }

    Head      head;
    Tail      tail;
};

template<template<int> typename CT,int end>
class RangeOfClasses<CT,end,end> {
public:
    /**/      RangeOfClasses( const auto &...args ) {}
    void      for_each_item ( auto &&f ) const {}
};
