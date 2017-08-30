# <img src="https://preview.ibb.co/bWyzXQ/dmaps_logo.png" width="250px" />

**DMAPS** is a C++ powered Python library implementing the [diffusion maps](https://en.wikipedia.org/wiki/Diffusion_map) manifold learning algorithm. It provides 
fast multi-threaded calculations for distances matrices and diffusion coordinates.

## Prerequisites 

**DMAPS** includes all external dependencies in the code base (see [acknowledgements](#ack)).
To build and install **DMAPS** you need the following:

- A compiler with C++11 support
- CMake >= 2.8.12
- Python development libraries

## Installation 

Make sure CMake and Python development libraries are installed. On a Debian-based distributions,
you can install them using the apt package manager. 

```bash
$ sudo apt install cmake python-dev
```

With the prerequisites satisfied, all you need to do is clone the repository and pip install. 

```bash
$ git clone https://github.com/hsidky/dmaps.git
$ pip install ./dmaps
```
## Features 

**DMAPS** is designed to perform nonlinear dimensionality reduction of high-dimensional 
data sets using the diffusion maps [[1]](#ref1) algorithm. In particular, this implementation 
is geared towards the analysis of molecular trajectories as first described in Ref. [[2]](#ref2).
Various metrics are provided to compute the distance matrix, with the ability to save and load 
data from disk. Both standard and locally-scaled diffusion maps can be generated. For local 
scaling, values of the kernel bandwidth are calculated using the scheme in Ref. [[3]](#ref3). 
It is also possible to weight the kernel for biased input data. 

Calculations of the distance matrix and local scale estimates are accelerated using OpenMP
multi-threading. To improve performance for large datasets, **Spectra** is used to compute 
only the top *k* eigenvectors requested by the user, since the desired *k* is usually a 
small number. *Eigen* also provides SIMD instructions for efficient linear algebra operations.

## Examples 

Below is an example that demonstrates basic usage of **DMAPS** on the classic 
Swiss roll dataset. For more detailed examples see the examples folder. 

```python 
import dmaps
import numpy as np
import matplotlib.pyplot as plt

# Assume we have the following numpy arrays:
# coords contains the [n, 3] generated coordinates for the Swiss roll dataset.
# color contains the position of the points along the main dimension of the roll. 
dist = dmaps.DistanceMatrix(coords)
dist.compute(metric=dmaps.metrics.euclidean)

# Compute top three eigenvectors. 
# Here we assume a good value for the kernel bandwidth is known.
dmap = dmaps.DiffusionMap(dist)
dmap.set_kernel_bandwidth(3)
dmap.compute(3)

# Plot result. Scale by top eigenvector.
v = dmap.get_eigenvectors()
w = dmap.get_eigenvalues()
plt.scatter(v[:,1]/v[:,0], v[:,2]/v[:,0], c=color)
plt.xlabel('$\Psi_2$')
plt.ylabel('$\Psi_3$')
```

The above code produces the diffusion map below.
<img src="https://preview.ibb.co/gnG6Wk/diffswiss.png" alt="diffswiss" border="0" />

That's pretty much it! Be sure to take a look in the examples folder for more sophisticated
applications.


## <a name="ack"></a> Acknowledgements 

**DMAPS** makes use of the following open source libraries:

- Pybind11 - C++11 Python bindings (https://github.com/pybind/pybind11)
- Eigen - C++ linear algebra library (http://eigen.tuxfamily.org/)
- Spectra - C++ library for large scale eigenvalue problems (https://spectralib.org/)

## License 

**DMAPS** is provided under an MIT license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

## References 

<a name="ref1"></a>[1] Coifman, R. R., & Lafon, S. (2006). Appl. Comput. Harmon. Anal., 21(1), 5–30.

<a name="ref2"></a>[2] Ferguson, A. L., et al. (2010). PNAS, 107(31), 13597–602.

<a name="ref3"></a>[3] Zheng, W., Rohrdanz, M. a, & Clementi, C. (2013). J. Phys. Chem. B, 117(42), 12769–12776.