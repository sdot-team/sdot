#pragma once

#include <assert.h>

#define ERROR( MSG ) ( [&]() { std::cerr << __FILE__ << ":" << __LINE__ << ": ERROR: " << MSG << "\n"; assert( 0 ); } )()
#define TODO ( [&]() { std::cerr << __FILE__ << ":" << __LINE__ << ": TODO" << "\n"; assert( 0 ); } )()
