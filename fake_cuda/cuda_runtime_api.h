#pragma once
// Stub: just enough for CCCL version checks
#include <cstddef>

#ifndef CUDART_VERSION
#  define CUDART_VERSION 12000
#endif

using cudaError_t  = int;
using cudaStream_t = void*;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };

inline cudaError_t cudaMemcpy( void*, const void*, size_t, cudaMemcpyKind ) { return 0; }
