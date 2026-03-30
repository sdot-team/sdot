#pragma once

#include <assert.h>

#define ERROR( MSG ) ( [&]() { std::cerr << __FILE__ << ":" << __LINE__ << ": ERROR: " << MSG << "\n"; exit( 1 ); } )()
#define TODO ( [&]() { std::cerr << __FILE__ << ":" << __LINE__ << ": TODO" << "\n"; exit( 1 ); } )()
