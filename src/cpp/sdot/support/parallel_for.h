#pragma once
#include <condition_variable>
#include <functional>
#include <algorithm>
#include <exception>
#include <cstddef>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

namespace sdot {

class ThreadPool {
public:
    explicit ThreadPool( unsigned int n ) : active_( 0 ), done_( false ), stop_( false ) {
        for( unsigned int i = 0; i < n; ++i )
            workers_.emplace_back( [this] { worker_loop(); } );
    }

    ~ThreadPool() {
        { std::unique_lock lock( mu_ ); stop_ = true; }
        cv_work_.notify_all();
        for( auto &t : workers_ ) t.join();
    }

    // Run func(i) for i in [start, end) in parallel, block until done.
    // Propagates the first exception thrown by any worker back to the caller.
    template<class Index, class F>
    void run( Index start, Index end, F &&func ) {
        const Index count = end - start;
        if ( count <= 0 ) return;

        const unsigned int n = static_cast<unsigned int>( workers_.size() );
        const Index chunk = ( count + n - 1 ) / n;

        exceptions_.assign( n, nullptr );
        active_.store( n );
        done_.store( false );

        {
            std::unique_lock lock( mu_ );
            for( unsigned int t = 0; t < n; ++t ) {
                const Index cs = start + static_cast<Index>( t ) * chunk;
                const Index ce = std::min( cs + chunk, end );
                tasks_.push_back( { [&func, &ex = exceptions_[ t ], cs, ce]() {
                    try {
                        for( auto i = cs; i < ce; ++i ) func( i );
                    } catch( ... ) {
                        ex = std::current_exception();
                    }
                } } );
            }
        }
        cv_work_.notify_all();

        // wait for completion
        { std::unique_lock lock( mu_done_ ); cv_done_.wait( lock, [&]{ return done_.load(); } ); }

        for( auto &ex : exceptions_ )
            if ( ex ) std::rethrow_exception( ex );
    }

    static ThreadPool &global() {
        static ThreadPool pool( std::max( 1u, std::thread::hardware_concurrency() ) );
        return pool;
    }

private:
    struct Task { std::function<void()> fn; };

    void worker_loop() {
        while ( true ) {
            Task task;
            {
                std::unique_lock lock( mu_ );
                cv_work_.wait( lock, [&]{ return stop_ || !tasks_.empty(); } );
                if ( stop_ && tasks_.empty() ) return;
                task = std::move( tasks_.back() );
                tasks_.pop_back();
            }
            task.fn();
            if ( active_.fetch_sub( 1 ) == 1 ) {
                std::unique_lock lock( mu_done_ );
                done_.store( true );
                cv_done_.notify_one();
            }
        }
    }

    std::vector<std::thread>         workers_;
    std::vector<Task>                tasks_;
    std::vector<std::exception_ptr>  exceptions_;
    std::mutex                       mu_, mu_done_;
    std::condition_variable          cv_work_, cv_done_;
    std::atomic<unsigned int>        active_;
    std::atomic<bool>                done_;
    bool                             stop_;
};


template<typename Index>
void parallel_for( Index start, Index end, auto &&func ) {
    const Index count = end - start;
    if ( count <= 1 ) {
        for( Index i = start; i < end; ++i )
            func( i );
        return;
    }

    ThreadPool::global().run( start, end, func );
}

} // namespace sdot
