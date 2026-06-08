#pragma once

/// Reusable host-side launcher for Metal compute kernels.
///
/// Included from a generated Objective-C++ (.mm) XLA FFI handler. The MSL kernel source is
/// compiled at runtime (newLibraryWithSource) and cached, so no offline metallib build is
/// needed. Buffers are created from the host (unified-memory) pointers the FFI handler
/// receives; buffers flagged as outputs are copied back after the command buffer completes.
///
/// Forward-only prototype path: flat 1D dispatch, FP32, no threadgroup tuning yet.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <initializer_list>
#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <map>

namespace sdot {

/// One buffer of a Metal launch: a host pointer, its byte size, and whether the kernel writes it.
struct MetalBuf {
    void  *ptr;        ///< host pointer (unified memory)
    size_t bytes;      ///< size in bytes
    bool   is_output;  ///< if true, copied back to `ptr` after the kernel completes
};

namespace metal_detail {
    inline id<MTLDevice> device() {
        static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev;
    }

    inline id<MTLCommandQueue> queue() {
        static id<MTLCommandQueue> q = [device() newCommandQueue];
        return q;
    }

    /// Compute pipeline state for (source, kernel name), compiled once and cached.
    inline id<MTLComputePipelineState> pipeline( const char *msl_source, const char *kernel_name ) {
        static std::map<std::string, id<MTLComputePipelineState>> cache;

        std::string key = kernel_name;
        key += '\n';
        key += msl_source;

        auto it = cache.find( key );
        if ( it != cache.end() )
            return it->second;

        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:msl_source];
        id<MTLLibrary> lib = [device() newLibraryWithSource:src options:nil error:&err];
        if ( !lib ) {
            fprintf( stderr, "[sdot] Metal library compilation failed: %s\n",
                     err ? [[err localizedDescription] UTF8String] : "?" );
            cache[ key ] = nil;
            return nil;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
        id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&err];
        if ( !pso )
            fprintf( stderr, "[sdot] Metal pipeline creation failed: %s\n",
                     err ? [[err localizedDescription] UTF8String] : "?" );

        cache[ key ] = pso;
        return pso;
    }
}

/// Launch `kernel_name` (defined in `msl_source`) over a flat 1D grid of `n_threads` threads.
/// Buffers are bound at successive indices [[buffer(0)]], [[buffer(1)]], ...
inline void metal_launch_1d( const char *msl_source, const char *kernel_name,
                             std::initializer_list<MetalBuf> buffers, size_t n_threads ) {
    if ( n_threads == 0 )
        return;

    @autoreleasepool {
        id<MTLComputePipelineState> pso = metal_detail::pipeline( msl_source, kernel_name );
        if ( !pso )
            return;

        id<MTLDevice>                dev = metal_detail::device();
        id<MTLCommandBuffer>         cb  = [metal_detail::queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];

        std::vector<MetalBuf>      specs( buffers );
        std::vector<id<MTLBuffer>> mbufs;
        mbufs.reserve( specs.size() );
        for ( size_t i = 0; i < specs.size(); ++i ) {
            id<MTLBuffer> b = [dev newBufferWithBytes:specs[ i ].ptr
                                               length:specs[ i ].bytes
                                              options:MTLResourceStorageModeShared];
            mbufs.push_back( b );
            [enc setBuffer:b offset:0 atIndex:i];
        }

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if ( tg > n_threads )
            tg = n_threads;
        [enc dispatchThreads:MTLSizeMake( n_threads, 1, 1 )
       threadsPerThreadgroup:MTLSizeMake( tg, 1, 1 )];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        for ( size_t i = 0; i < specs.size(); ++i )
            if ( specs[ i ].is_output )
                std::memcpy( specs[ i ].ptr, [mbufs[ i ] contents], specs[ i ].bytes );
    }
}

} // namespace sdot
