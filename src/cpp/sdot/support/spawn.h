#include <tl/support/containers/Vec.h>
#include <functional>
#include <thread>

namespace sdot {

inline void spawn( const std::function<void( int num_thread, int nb_threads )> &f, PI nb_threads ) {
    // launch
    Vec<std::thread> threads( FromSizeAndInitFunctionOnIndex(), nb_threads, [&]( std::thread *th, PI num_thread ) {
        new ( th ) std::thread( [num_thread,nb_threads,&f]() {
            f( num_thread, nb_threads );
        } );
    } );

    // join
    for( std::thread &th : threads )
        th.join();
}

} // namespace sdot
