#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

#include "cubature.hpp"

int main() {
    double a = 0.7;
    double b = 1.1;

    auto func = [&a,&b] (double x, double y) { return std::exp(-a*x*x) * std::exp(-b*y*y); };
    
    // Timing
    {
        auto start = std::chrono::high_resolution_clock::now();

        // non-vectorized
        cubature::integrate<false>(func,{-INFINITY,-INFINITY},{+INFINITY,+INFINITY});

        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Time (non-vectorized): " << std::chrono::duration<double>{stop - start}.count() << " sec\n";
    }
    {
        auto start = std::chrono::high_resolution_clock::now();

        // vectorized
        cubature::integrate<true>(func,{-INFINITY,-INFINITY},{+INFINITY,+INFINITY});

        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Time (vectorized)    : " << std::chrono::duration<double>{stop - start}.count() << " sec\n";
    }

    // Accuracy
    {
        auto result = cubature::integrate<true>(func,{-INFINITY,-INFINITY},{+INFINITY,+INFINITY});

        double exact = M_PI/std::sqrt(a*b);
    
        std::cout.precision(std::numeric_limits<double>::digits10);
        std::cout << "Result          = " << std::get<0>(result) << '\n'
                  << "Numerical error = " << std::get<1>(result) << '\n';
        std::cout << "Exact           = " << exact << '\n'
                  << "Actual error    = " << std::get<0>(result)-exact << '\n';
    }

    // Complex integrand
    {
        constexpr std::complex<double> const I(0,1);

        auto complex_func = [&I] (double x, double y, double kx, double ky) {
            auto value = (std::cos(kx/2) + I*std::sin(ky/2)) * std::exp(-x*x) *std::exp(-y*y);
            return std::vector<double>{std::real(value), std::imag(value)};
        };
        auto result = cubature::integrate<true>(complex_func, 2, {-INFINITY,-INFINITY,0,0},{+INFINITY,+INFINITY,2*M_PI,2*M_PI});

        std::complex<double> exact = 8.0 * I * M_PI*M_PI;

        std::complex<double> complex_result{std::get<0>(result)[0], std::get<0>(result)[1]};
        std::complex<double> complex_error{std::get<1>(result)[0], std::get<1>(result)[1]};
        std::cout << "Result          = " << complex_result << '\n'
                  << "Numerical error = " << complex_error << '\n';
        std::cout << "Exact           = " << exact << '\n'
                  << "Actual error    = " << complex_result-exact << '\n';
    }

}
