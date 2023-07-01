## Numpy bfloat16 and posit8 datatypes

Based directly on the [`bfloat16`](https://github.com/GreenWaves-Technologies/bfloat16) **numpy** datatype wrapper
by _Greenwave Technologies_, this repository extends it to add a posit datatype
wrapper for **numpy**.

Whereas _Greenwave Technologies_ used **eigen** library for the datatype
backend, I have used the [Stillwater Supercomputing's](https://github.com/stillwater-sc/universal) posit dataype implementation.

## How to build the repository

### Legacy build method
The **setup.cfg** file follows [this Python tutorial](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html).
The custom location is provided using [this Python tutorial](https://setuptools.pypa.io/en/latest/deprecated/easy_install.html#custom-installation-locations).



### New build method
The method for building the project follows from [this blog](https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html#summary),
the [Python packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) and the [pypa setuptools tutorial](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html). 




## How to use the datatype

