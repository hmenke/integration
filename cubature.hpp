#pragma once

#include <array>
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

    F m_f;

    template < std::size_t... I >
    static int cubature_wrapper_detail(unsigned /* ndim */, size_t npts, const double *x, void *fdata,
                                       unsigned /* fdim */, double *fval, index_sequence<I...>) {
        cubature_impl * t = static_cast<cubature_impl*>(fdata);

        #pragma omp parallel for
        for (size_t j = 0; j < npts; ++j) { // evaluate the integrand for npts points
            double dt = 1;

            // "expander trick"
            using expander = int[];
            (void) expander { 0, ( (void) (
                                       dt *= (1+x[j*m_dim+I]*x[j*m_dim+I])
                                       /
                                       ((1-x[j*m_dim+I]*x[j*m_dim+I])*(1-x[j*m_dim+I]*x[j*m_dim+I]))
                                       ),0) ... };

            fval[j] = t->m_f((x[j*m_dim+I]/(1-x[j*m_dim+I]*x[j*m_dim+I]))...) * dt;
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
    cubature_impl(F f) : m_f{f} {}

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
    integrate(double epsabs, double epsrel, unsigned limit) {
        integrand_t Fint = &cubature_wrapper;

        std::array<double,m_dim> min; min.fill(-1);
        std::array<double,m_dim> max; max.fill(+1);
        
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

template < bool vectorize = false, typename F, unsigned dim = detail::argument_count<F>::value >
std::tuple<double,double>
integrate_i(F func,
            double epsabs = 1.49e-8, double epsrel = 1.49e-8,
            unsigned limit = 0) {
    return detail::cubature_impl<vectorize,F,dim>(func).integrate(epsabs, epsrel, limit);
}

} // namespace cubature
