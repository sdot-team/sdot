#include <tl/support/containers/Vec.h>
#include <functional>
#include <thread>

namespace sdot {

inline void spawn( const std::function<void( int num_thread, int nb_threads )> &f, PI nb_threads ) {
    // special case: no need to create thread(s)
    if ( nb_threads == 1 )
        return f( 0, 1 );

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
