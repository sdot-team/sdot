#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>
#include <thread>

namespace sdot {

/// Pool of persistent threads, reused from one job to the next (no re-creation).
///
/// Threads are created once (at construction) and wait on `cv_work`. `run`
/// publishes a function to execute and wakes everyone up; each worker runs it in
/// parallel then goes back to sleep. `run` returns once every requested thread is
/// done.
///
/// Three counters, not to be confused:
///   - `pool_size`      : total number of threads in the pool, fixed at construction.
///   - `job_nb_threads` : number of threads requested for the current job (<= pool_size);
///                        this is the value passed to the callback as `nb_threads`.
///   - `nb_busy`        : number of job threads not yet done; countdown decremented by
///                        each worker, reaches 0 => job finished.
struct WaitingThreads {
    using                    Func              = std::function<void( int num_thread, int nb_threads )>;

    /**/                     WaitingThreads    ();
    /**/                    ~WaitingThreads    ();

    /// Run `func` on `max_nb_threads` threads (whole pool if < 0 or too large) and
    /// wait for all of them to finish before returning.
    void                     run               ( const Func &func, int max_nb_threads = -1 );

    const Func              *func_to_execute;  ///< current job (only valid during `run`)
    std::vector<std::thread> threads;          ///< the `pool_size` workers
    std::condition_variable  cv_work;          ///< wakes the workers when a job is published
    std::condition_variable  cv_done;          ///< wakes `run` when `nb_busy` reaches 0
    std::mutex               mutex;
    int                      job_nb_threads;   ///< nb of threads requested for the current job
    int                      work_version;     ///< bumped on each job, wakes the workers
    int                      pool_size;        ///< total nb of threads (see doc above)
    int                      nb_busy;          ///< nb of job threads not yet done
    bool                     stop;             ///< shutdown request (destruction)

private:
    void worker( int num_thread );
};

/// Global pool, shared by the whole process (created on first call).
WaitingThreads &waiting_threads();

} // namespace sdot

#include "WaitingThreads.cxx" // IWYU pragma: export
