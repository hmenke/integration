# Integration

[![Build status][travis-svg]][travis-link]

This project enables multidimensional integration in C++ using the
[cubature](https://github.com/stevengj/cubature) library.  The
implementation is located in `cubature.hpp` and a sample of usage is
provided together with integration into the CMake build system.

# Documentation

**Note:** Currently the integration is limited to [-∞,∞] in all
dimensions.  No other limits are possible at the moment.  (That's
also why the function is called `integrate_i`, with `i` for infinity)

The function signature of the public `integrate_i` function is
```cpp
template < bool vectorize = false, typename F, unsigned dim = detail::argument_count<F>::value >
std::tuple<double,double>
integrate_i(F func,
            double epsabs = 1.49e-8, double epsrel = 1.49e-8,
            unsigned limit = 0)
```
The first template parameter `vectorize` determines whether to use
OpenMP for parallel evaluation of the function and possibly much
faster integration.  The other template parameters are deduced using
template argument deduction and are of no interest for the user.

The first parameter of the function is the function to integrate.
This can be either a function pointer, a lambda function, or a functor
with overloaded call operator.  The dimensionality of the integral is
deduced from the number of arguments of said function.  For example if
`func` was
```cpp
double func(double x, double y, double z)
```
the integral would be over all three dimensions.  If you have to pass
additional parameters to the function, use a lambda or a functor.

The parameters `epsabs` and `epsrel` determine the absolute and
relative accuracy of the integral which gives the convergence
criterion for the integrator.  Don't choose this to small or you'll
wait for ages.  The final parameter `limit` gives a bound on the
maximum number of function evaluations which are allowed for the
integrator.  Setting this to zero means no limit.

# Requirements

* C++11 capable compiler
* OpenMP (only if you want to parallelize, otherwise the compiler will just ignore the `#pragma`)

[travis-svg]: https://travis-ci.org/hmenke/integration.svg?branch=master
[travis-link]: https://travis-ci.org/hmenke/integration
