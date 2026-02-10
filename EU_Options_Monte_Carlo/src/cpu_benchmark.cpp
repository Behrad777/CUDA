// src/cpu_benchmark.cpp
#include "cpu_benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <random>

// Simple single-thread CPU Monte Carlo baseline for a European call under GBM.
// Uses standard normal draws and discounts by exp(-rT).
double cpu_monte_carlo_call_price_timed(const OptionParams& p,
                                        int num_paths,
                                        std::uint64_t seed,
                                        double* loop_ms_out)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const double S0    = static_cast<double>(p.S0);
    const double K     = static_cast<double>(p.K);
    const double r     = static_cast<double>(p.r);
    const double sigma = static_cast<double>(p.sigma);
    const double T     = static_cast<double>(p.T);

    // Precompute drift and volatility terms
    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double vol   = sigma * std::sqrt(T);

    double sum_payoff = 0.0;

    const auto loop_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_paths; ++i) {
        const double z  = norm(rng);                       // Z ~ N(0,1)
        const double ST = S0 * std::exp(drift + vol * z);  // terminal price
        const double payoff = std::max(ST - K, 0.0);       // call payoff
        sum_payoff += payoff;
    }
    const auto loop_end = std::chrono::high_resolution_clock::now();
    if (loop_ms_out) {
        *loop_ms_out =
            std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
    }

    const double mean_payoff = sum_payoff / static_cast<double>(num_paths);
    const double discounted  = std::exp(-r * T) * mean_payoff;

    return discounted;
}

double cpu_monte_carlo_call_price(const OptionParams& p, int num_paths, std::uint64_t seed)
{
    return cpu_monte_carlo_call_price_timed(p, num_paths, seed, nullptr);
}
