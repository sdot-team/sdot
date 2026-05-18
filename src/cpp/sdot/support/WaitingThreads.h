#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>
#include <thread>

namespace sdot {

struct WaitingThreads {
    /**/ WaitingThreads() : func_to_execute( nullptr ), nb_active( 0 ), active_nb_threads( 0 ), work_version( 0 ), stop( false ) {
        nb_threads = (int)std::max( 1u, std::thread::hardware_concurrency() );
        threads.resize( nb_threads );
        for ( int i = 0; i < nb_threads; ++i )
            threads[ i ] = std::thread( &WaitingThreads::worker, this, i );
    }

    /**/ ~WaitingThreads() {
        { std::unique_lock<std::mutex> lock( mutex ); stop = true; }
        cv_work.notify_all();
        for ( auto &t : threads )
            t.join();
    }

    void run( const std::function<void( int num_thread )> &func, int max_nb_threads = -1 ) {
        if ( max_nb_threads < 0 )
            max_nb_threads = nb_threads;

        {
            std::unique_lock<std::mutex> lock( mutex );
            func_to_execute   = &func;
            active_nb_threads = max_nb_threads;
            nb_active         = max_nb_threads;
            ++work_version;
        }
        cv_work.notify_all();

        std::unique_lock<std::mutex> lock( mutex );
        cv_done.wait( lock, [&]{ return nb_active == 0; } );
        func_to_execute = nullptr;
    }

    void worker( int num_thread ) {
        int seen_version = 0;
        while ( true ) {
            std::unique_lock<std::mutex> lock( mutex );
            cv_work.wait( lock, [&]{ return stop || work_version != seen_version; } );
            if ( stop ) break;
            seen_version  = work_version;
            if ( num_thread >= active_nb_threads ) continue;
            auto *f       = func_to_execute;
            lock.unlock();

            (*f)( num_thread );

            lock.lock();
            if ( --nb_active == 0 )
                cv_done.notify_one();
        }
    }

    const std::function<void( int num_thread )> *func_to_execute;
    std::vector<std::thread>                     threads;
    std::condition_variable                      cv_work;
    std::condition_variable                      cv_done;
    std::mutex                                   mutex;
    int                                          nb_threads;
    int                                          nb_active;
    int                                          active_nb_threads;
    int                                          work_version;
    bool                                         stop;
};

WaitingThreads &waiting_threads() {
    static WaitingThreads res;
    return res;
}

} // namespace sdot
