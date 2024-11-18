SDOT
====

This package contains tool to handle
* semi-discrete transport plans (between discrete and generic densities),
* polyhedral convex functions (represented as a max of affine functions),
* and power diagrams (that can be seen as a generalization of Voronoi diagrams).

It works in any number of dimensions, and is highly optimized, in terms of execution speed and memory usage.

Historically, this package is a re-design of the `pysdot` package, which was written to handle a wide number of semi-discrete transport applications (partial transport, Moreau-Yosida regularization, etc...) but only worked in 2D or 3D. Additionaly, and more specifically, we wanted to make operations on power diagram much more comprehensive and generic.

Currently, there are bindings C++ and Python, but more languages are in the pipeline (feel free to ask if your favorite language is not represented :) ).

Installation
------------

For python, `pip install sdot` should do the job, including some precompiled libraries for the most common cases (2D/3D, float64, ...). If your case is not included in the distribution, the required dynamic libraries will be automatically compiled on first use. For this case, you will need to have a C++ compiler installed on your machine. As scons is used to find and call the compiler, all you need to do is install one compiler that is compatible with this builder (for instance g++, clang, xcode, vscode, ...). 

To get that latest version, `sdot` can also be installed from the git repository.

For the python modules:

```bash
git clone https://github.com/sdot-team/sdot.git
# maybe after a micromamba activate ...
cd sdot/src/python
pip install flit
flit install -s # -s makes symbolic links to the sources
```

Modules
-------

* [Optimal transport operations in python](doc/optimal_transport_py.md).
* [Power or voronoi diagrams and cells in python](doc/power_diagram_py.md).
* [Polyhedral convex function in python](doc/polyhedral_convex_py.md).

A word on performance
---------------------

The most common tools to handle voronoi and power diagrams start from delaunay (regular) triangulations. Building this triangulation is generaly the most time-consuming part of this approach, notably because one have to deal with the problems that come with digital precision...

Nevertheless, for most applications of the sdot package, we only need the *integrals* of the cells and the boundaries, meaning that most of the problems with digital precision naturally vanish at the end. We also realized that it was much more convenient for the user to work directly with the cells.

Bearing in mind that exact connectivity may of course be required for some applications (the 'dual' geometry becoming the triangulation in our case :) ), we therefore decided to give a try to algorithms with a focus on the cells, that are computed individually in a fully parallel fashion. We tried to stay on the user specified scalar types (e.g. `float64`) as long as possible. Finally, we designed adapted spatial acceleration structures to stay within O(n log(n)) execution speed with the smallest possible constant.

Internally, it is written in C++/Cuda with SIMD/SIMT instructions. It support large vectors, for out-of-core and multi-machine computations. More on [this page](doc/performance.md).

On going work
-------------

* pytorch compatible operations
* pre-guess of the weight to avoid void cells at the beginning
* non-linear solvers to avoid bad newton directions
