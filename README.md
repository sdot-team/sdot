SDOT
====

This package contains tools to handle
* semi-discrete transport plans (between discrete and generic densities),
* polyhedral convex functions (max of affine functions),
* and power diagrams (a generalization of Voronoi diagrams).

It works in any number of dimensions. It is highly optimized, in terms of execution speed and memory usage.

Historically, this package is a re-design of the `pysdot` package, which was written to handle a wide number of semi-discrete transport applications (partial transport, Moreau-Yosida regularization, etc...) but only worked in 2D or 3D. Additionaly, and more specifically, we wanted the APIs much more comprehensive and generic.

Currently, there are bindings for C++ and Python.

Installation
------------

### Pip

For python, `pip install sdot` should do the job, including some precompiled libraries for the most common cases (2D/3D, float64, ...). If your cases are not included in the distribution, the required dynamic libraries will be automatically compiled on first use. In this situation, you will need to have a C++ compiler installed on your machine. As scons is used to find and call the compiler, all you need to do is install one compiler that is compatible with this builder (for instance g++, clang, xcode, vscode, ...). 

For a compiler : under Debian like, `sudo apt install g++`. Under Mac os, `xcode-select --install`. Under Windows, you can follow [this link](https://code.visualstudio.com/docs/cpp/config-mingw#_prerequisites).

### Sources

To get that latest version, `sdot` can also be installed from the git repository.

For the python modules:

```bash
git clone https://github.com/sdot-team/sdot.git
# maybe after a micromamba activate ...
cd sdot/src/python
pip install flit
flit install -s # -s makes symbolic links to the sources
```

Notebook examples
-----------------

Here are some notebooks you can download and test on your own machine to understand the overall spirit.

* Power or voronoi diagrams in python: [file](examples/tutorials/0_power_diagram_py.ipynb), [colab](https://colab.research.google.com/drive/1yT62po-HFCxeXD4D_6XF8pHMgl3Fut34?usp=sharing).
* Optimal transport operations in python: [file](examples/tutorials/1_optimal_transport_py.ipynb), [colab](https://colab.research.google.com/drive/1P7l7_8QaEUFiVz49Ll1Avhwi8meaHojl?usp=sharing).
* Polyhedral convex function in python: [file](examples/tutorials/2_polyhedral_convex_py.ipynb), [colab](...).

If you're looking for more simple examples, there is an [example directory](examples/) with more concise notebooks, oriented on specific tasks.

Extensive documentation
-----------------------

[The generated pydoc files](...)

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
