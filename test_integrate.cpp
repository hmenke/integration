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
        cubature::integrate_i<false>(func);

        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Time (non-vectorized): " << std::chrono::duration<double>{stop - start}.count() << " sec\n";
    }
    {
        auto start = std::chrono::high_resolution_clock::now();

        // vectorized
        cubature::integrate_i<true>(func);

        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Time (vectorized)    : " << std::chrono::duration<double>{stop - start}.count() << " sec\n";
    }

    // Accuracy
    {
        auto result = cubature::integrate_i<true>(func);

        double exact = M_PI/std::sqrt(a*b);
    
        std::cout.precision(std::numeric_limits<double>::digits10);
        std::cout << "Result          = " << std::get<0>(result) << '\n'
                  << "Numerical error = " << std::get<1>(result) << '\n';
        std::cout << "Exact           = " << exact << '\n'
                  << "Actual error    = " << std::abs(std::get<0>(result)-exact) << '\n';
    }

    constexpr std::complex<double> const I(0,1);
    auto complex_func = [&I] (double x, double y) { return (std::cos(x) + I*std::sin(y)) * std::exp(-x*x) *std::exp(-y*y); };
    auto result = cubature::integrate_i<true>(complex_func);
    std::cout << "Result          = " << std::get<0>(result) << '\n'
              << "Numerical error = " << std::get<1>(result) << '\n';

}
