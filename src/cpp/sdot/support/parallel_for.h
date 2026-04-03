#pragma once
#include <thread>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <exception>

namespace sdot {

// Simple parallel_for using std::thread (no dependencies, C++11)
// Automatically detects CPU count and splits work
template<typename Index>
void parallel_for( Index start, Index end, auto &&func ) {
    if ( start >= end )
        return;

    const Index count = end - start;

    // Get number of threads (default to hardware concurrency)
    const unsigned int num_threads = std::max(1u, std::min(
        std::thread::hardware_concurrency(),
        static_cast<unsigned int>(count)
    ));

    // Serial execution for small workloads
    if ( count <= 1 || num_threads == 1 ) {
        for( Index i = start; i < end; ++i )
            func( i );
        return;
    }

    // Parallel execution — propagate exceptions back to the calling thread
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions( num_threads );
    threads.reserve( num_threads );

    const Index chunk_size = ( count + num_threads - 1 ) / num_threads;

    for( unsigned int t = 0; t < num_threads; ++t ) {
        const Index chunk_start = start + t * chunk_size;
        const Index chunk_end = std::min( chunk_start + chunk_size, end );

        if ( chunk_start >= chunk_end )
            break;

        threads.emplace_back( [ chunk_start, chunk_end, &func, &ex = exceptions[ t ] ]() {
            try {
                for( Index i = chunk_start; i < chunk_end; ++i )
                    func( i );
            } catch ( ... ) {
                ex = std::current_exception();
            }
        });
    }

    for( auto &thread : threads )
        thread.join();

    for( auto &ex : exceptions )
        if ( ex )
            std::rethrow_exception( ex );
}

} // namespace sdot
