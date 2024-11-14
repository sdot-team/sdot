SDOT
====

This package contains tool to handle
* semi-discrete transport plans (between discrete and generic densities),
* polyhedral convex functions (represented as a max of affine functions),
* power diagrams (that can be seen as a generalization of Voronoi diagrams).

It is highly optimized, in terms of execution speed and memory usage (more on [this page](doc/performance.md)).

It works in any number of dimensions (>= 1).

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

