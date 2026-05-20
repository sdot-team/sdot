#pragma once

#include "WaitingThreads.h" // IWYU pragma: keep

namespace sdot {

inline WaitingThreads::WaitingThreads() : func_to_execute( nullptr ), job_nb_threads( 0 ), work_version( 0 ), nb_busy( 0 ), stop( false ) {
    pool_size = (int)std::max( 1u, std::thread::hardware_concurrency() );
    threads.resize( pool_size );
    for ( int i = 0; i < pool_size; ++i )
        threads[ i ] = std::thread( &WaitingThreads::worker, this, i );
}

inline WaitingThreads::~WaitingThreads() {
    { std::unique_lock<std::mutex> lock( mutex ); stop = true; }
    cv_work.notify_all();
    for ( auto &t : threads )
        t.join();
}

inline void WaitingThreads::run( const Func &func, int max_nb_threads ) {
    if ( max_nb_threads < 0 || max_nb_threads > pool_size )
        max_nb_threads = pool_size;

    // publish the job and wake the workers
    {
        std::unique_lock<std::mutex> lock( mutex );
        job_nb_threads  = max_nb_threads;
        nb_busy         = max_nb_threads;
        func_to_execute = &func;
        ++work_version;
    }
    cv_work.notify_all();

    // wait for every requested thread to finish
    std::unique_lock<std::mutex> lock( mutex );
    cv_done.wait( lock, [&]{ return nb_busy == 0; } );
    func_to_execute = nullptr;
}

inline void WaitingThreads::worker( int num_thread ) {
    int seen_version = 0;
    while ( true ) {
        std::unique_lock<std::mutex> lock( mutex );
        cv_work.wait( lock, [&]{ return stop || work_version != seen_version; } );
        if ( stop )
            break;

        seen_version = work_version;

        // this worker does not take part in the current job
        if ( num_thread >= job_nb_threads )
            continue;

        auto *f = func_to_execute;
        lock.unlock();

        ( *f )( num_thread, job_nb_threads );

        lock.lock();
        if ( --nb_busy == 0 )
            cv_done.notify_one();
    }
}

inline WaitingThreads &waiting_threads() {
    static WaitingThreads res;
    return res;
}

} // namespace sdot
