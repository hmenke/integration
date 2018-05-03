#pragma once

#include <array>
#include <complex>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <cubature/cubature.h>

namespace cubature {

namespace detail {

// argument_count

template < unsigned v >
using unsigned_constant = std::integral_constant<unsigned, v>;

template < typename F >
struct argument_count : argument_count<decltype(&F::operator())> {};

template <typename R, typename C, typename... Args>
struct argument_count<R(C::*)(Args...)> : unsigned_constant<sizeof...(Args)> {};

template <typename R, typename C, typename... Args>
struct argument_count<R(C::*)(Args...) const> : unsigned_constant<sizeof...(Args)> {};

template <typename R, typename... Args>
struct argument_count<R(Args...)> : unsigned_constant<sizeof...(Args)> {};

template <typename R, typename... Args>
struct argument_count<R(*)(Args...)> : unsigned_constant<sizeof...(Args)> {};

// Return type deduction

template < typename F >
struct return_of : return_of<decltype(&F::operator())> {};

template <typename R, typename C, typename... Args>
struct return_of<R(C::*)(Args...)> { typedef R type; };

template <typename R, typename C, typename... Args>
struct return_of<R(C::*)(Args...) const> { typedef R type; };

template <typename R, typename... Args>
struct return_of<R(Args...)> { typedef R type; };

template <typename R, typename... Args>
struct return_of<R(*)(Args...)> { typedef R type; };

// is_complex

template < typename T >
struct is_complex : std::false_type {};

template < typename T >
struct is_complex<std::complex<T>> : std::true_type { typedef T value_type; };

// real_imag

template < typename F, typename G >
struct function_composition {
    F f;
    G g;

    function_composition(F f, G g) : f(f), g(g) {}

    template < typename ... Args >
    auto operator()(Args ... args) -> decltype(f(g(std::declval<Args>()...))) const {
        return f(g(args...));
    }
};

template < typename F, typename G >
function_composition<F,G> make_function_composition(F f, G g) {
    return function_composition<F,G>(f,g);
}

// index_sequence

template <size_t ...I>
struct index_sequence {};

template <size_t N, size_t ...I>
struct make_index_sequence : public make_index_sequence<N - 1, N - 1, I...> {};

template <size_t ...I>
struct make_index_sequence<0, I...> : public index_sequence<I...> {};

// cubature_impl

template < bool vectorize, typename F, unsigned m_dim >
class cubature_impl
{
    using integrand_t = typename std::conditional<vectorize,integrand_v,integrand>::type;

    enum limits {
        INFINITE,
        HALF_INF_LOWER,
        HALF_INF_UPPER,
        FINITE
    };

    std::array<limits,m_dim> isfinite;
    F m_f;

    template < std::size_t... I >
    static int cubature_wrapper_detail(unsigned /* ndim */, size_t npts, const double *x, void *fdata,
                                       unsigned /* fdim */, double *fval, index_sequence<I...>) {
        cubature_impl * p = static_cast<cubature_impl*>(fdata);

        // evaluate the integrand for npts points
        #pragma omp parallel for if(npts > 1)
        for (size_t j = 0; j < npts; ++j) {
            double t[m_dim] = { 0 };  // initializer is needed to make compiler happy
            double dt = 1.0;

            for (size_t i = 0; i < m_dim; ++i) {
                switch (p->isfinite[i]) {
                case FINITE:
                    t[i] = x[j*m_dim+i];
                    break;
                case HALF_INF_LOWER:
                    // To be implemented
                    break;
                case HALF_INF_UPPER:
                    // To be implemented
                    break;
                case INFINITE:
                    t[i] = x[j*m_dim+i]/(1-x[j*m_dim+i]*x[j*m_dim+i]);
                    dt *= (1+x[j*m_dim+i]*x[j*m_dim+i])
                        /
                        ((1-x[j*m_dim+i]*x[j*m_dim+i])*(1-x[j*m_dim+i]*x[j*m_dim+i]));
                    break;
                }
            }

            fval[j] = p->m_f(t[I]...) * dt;
        }

        return 0; // success
    }

    template < bool enable = !vectorize >
    static typename std::enable_if< enable, int >::type
    cubature_wrapper(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
        return cubature_wrapper_detail(ndim, 1, x, fdata, fdim, fval, make_index_sequence<m_dim>{});
    }

    template < bool enable = vectorize >
    static typename std::enable_if< enable, int >::type
    cubature_wrapper(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval) {
        return cubature_wrapper_detail(ndim, npts, x, fdata, fdim, fval, make_index_sequence<m_dim>{});
    }

public:
    cubature_impl(F f) : isfinite{}, m_f{f} {}

    template < bool enable = !vectorize >
    typename std::enable_if< enable, int >::type
    hcubature_dispatch(unsigned fdim, integrand f, void *fdata,
                       unsigned dim, const double *xmin, const double *xmax,
                       unsigned maxEval, double reqAbsError, double reqRelError,
                       error_norm norm, double *val, double *err) {
        return hcubature(fdim,f,fdata,dim,xmin,xmax,maxEval,reqAbsError,reqRelError,norm,val,err);
    }

    template < bool enable = vectorize >
    typename std::enable_if< enable, int >::type
    hcubature_dispatch(unsigned fdim, integrand_v f, void *fdata,
                       unsigned dim, const double *xmin, const double *xmax,
                       unsigned maxEval, double reqAbsError, double reqRelError,
                       error_norm norm, double *val, double *err) {
        return hcubature_v(fdim,f,fdata,dim,xmin,xmax,maxEval,reqAbsError,reqRelError,norm,val,err);
    }

    std::tuple<double,double>
    integrate(std::array<double,m_dim> min,
              std::array<double,m_dim> max,
              double epsabs, double epsrel, unsigned limit) {
        integrand_t Fint = &cubature_wrapper;

        for (size_t i = 0; i < m_dim; ++i) {
            if (std::isinf(min[i]) && std::isinf(max[i])) {
                min[i] = -1;
                max[i] = +1;
                isfinite[i] = INFINITE;
            } else if (std::isinf(min[i]) || std::isinf(max[i])) {
                throw std::runtime_error("Half-infinite intervals are not yet supported!");
            } else {
                isfinite[i] = FINITE;
            }
        }
        
        double result, error;

        int status = hcubature_dispatch(
            /* fdim */ 1, /* integrand */ Fint, /* fdata */ this,
            /* dim */ m_dim, /* xmin */ min.data(), /* xmax */ max.data(),
            /* maxEval */ limit, /* reqAbsError */ epsabs, /* reqRelError */ epsrel,
            /* error_norm */ ERROR_INDIVIDUAL, /* val */ &result, /* err */ &error);

        if ( status != 0 ) {
            throw std::runtime_error("Error during integration!");
        }

        return std::make_tuple(result, error);
    }
};

} // namespace detail

// public integrate function

#ifndef CUBATURE_VECTORIZE_DEFAULT
#define CUBATURE_VECTORIZE_DEFAULT false
#endif

template < bool vectorize = CUBATURE_VECTORIZE_DEFAULT,
           int&... ExplicitArgumentBarrier,
           typename F,
           unsigned dim = detail::argument_count<F>::value,
           typename return_t = typename std::decay<typename detail::return_of<F>::type>::type
           >
typename std::enable_if<std::is_floating_point<return_t>::value,std::tuple<double,double>>::type
integrate(F func,
          std::array<double,dim> const &min, std::array<double,dim> const &max,
          double epsabs = 1.49e-8, double epsrel = 1.49e-8,
          unsigned limit = 0) {
    return detail::cubature_impl<vectorize,F,dim>(func).integrate(min, max, epsabs, epsrel, limit);
}

template < bool vectorize = CUBATURE_VECTORIZE_DEFAULT,
           int&... ExplicitArgumentBarrier,
           typename F,
           unsigned dim = detail::argument_count<F>::value,
           typename return_t = typename std::decay<typename detail::return_of<F>::type>::type
           >
typename std::enable_if<detail::is_complex<return_t>::value,std::tuple<std::complex<double>,std::complex<double>>>::type
integrate(F func,
          std::array<double,dim> const &min, std::array<double,dim> const &max,
          double epsabs = 1.49e-8, double epsrel = 1.49e-8,
          unsigned limit = 0) {
    constexpr std::complex<double> const I(0,1);
    using T = typename detail::is_complex<return_t>::value_type;
    auto real_part = detail::make_function_composition(static_cast<T(*)(std::complex<T> const &)>(std::real),func);
    auto imag_part = detail::make_function_composition(static_cast<T(*)(std::complex<T> const &)>(std::imag),func);
    auto real_res = detail::cubature_impl<vectorize,decltype(real_part),dim>(real_part).integrate(min, max, epsabs, epsrel, limit);
    auto imag_res = detail::cubature_impl<vectorize,decltype(imag_part),dim>(imag_part).integrate(min, max, epsabs, epsrel, limit);
    return std::make_tuple(std::get<0>(real_res) + I*std::get<0>(imag_res), std::get<1>(real_res) + I*std::get<1>(imag_res));
}

} // namespace cubature
