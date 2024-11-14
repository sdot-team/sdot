SDOT
====

This package contains tool to handle
* semi-discrete transport plans (between discrete and generic densities),
* polyhedral convex functions (represented as a max of affine functions),
* power diagrams (that can be seen as a generalization of Voronoi diagrams).

It works in any number of dimensions, with several types of scalar.

It is highly optimized, in terms of execution speed and memory usage (more on [this page](doc/performance.md)).

This package is a redesign of the pysdot package, which handled a number of semi-discrete transport applications (partial transport, Moreau-Yosida regularization, etc.) but only worked in 2D or 3D. In addition, we wanted to make power diagram operations much more comprehensive and generic.

Currently, there are bindings C++ and Python, but more languages are in the pipeline (feel free to ask if your favorite language is not represented :) ).

Installation
------------

For python, `pip install sdot` will install the package, with precompiled libraries for the most common cases (2D/3D, float64, ...). If your case is not included in the distribution, libraries will be automatically compiled on first use. In this case, you will need to have a C++ compiler installed on your machine. As scons is used to find the compiler, all you need to do is install one that is compatible with this builder (g++, clang, xcode, vscode, ...). 

Tutorials
---------

* [A first tutorial for optimal transport operations in python](doc/optimal_transport_py.md).
* [A tutorial for power/voronoi diagram and cells in python](doc/power_diagram_py.md).
* [A tutorial for polyhedral convex function in python](doc/polyhedral_convex_py.md).

A word on performance
---------------------

The most common tools to handle voronoi and power diagrams starts from delaunay (regular) triangulations. Building this triangulation is the most time-consuming part of this approach, notably because one have to deal with the problems that come with digital precision...

Nevertheless, for most applications of the sdot package, we only need the *integrals* of the cells and the boundaries, meaning that most of the problems with digital precision naturally vanish at this stage. We also realized that it was much more convenient for the user to work directly with the cells.

Bearing in mind that exact connectivity may of course be required for some applications (the 'dual' geometry becoming the triangulation in our case :) ), we therefore decided to give a try to algorithms where we focus on cells, that are computed individually in a fully parallel fashion. We tried to stay on the user specified scalar types as long as possible. Finally, we designed adapted spatial acceleration structures to stay within O(n log(n)) execution speed, with the smallest possible constant.

Internally, it is written in C++/Cuda with SIMD/SIMT instructions. It support large vector, for out-of-core and multi-machine computations.

