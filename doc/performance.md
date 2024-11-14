Execution speed and memory usage
================================

The most common tools to handle voronoi and power diagrams starts from delaunay (regular) triangulations. Building this triangulation is the most time-consuming part of this approach, notably because we have to deal with the problems that come with digital precision.

For most applications of the sdot package, we only need cell and boundary integrals, and it's more convenient for the user to work directly on cells. Bearing in mind that exact connectivity may be required for some applications (the dual becoming triangulation :) ), we therefore decided to try an algorithm where cells are computed individually, in a fully parallel fashion, with adapted spatial acceleration structures.

Internally, it is written in C++ and highly optimized (use of SIMD/SIMT instructions, O(N log N) algorithms where possible, ...).
