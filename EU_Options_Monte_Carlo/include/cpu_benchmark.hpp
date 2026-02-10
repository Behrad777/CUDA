#pragma once

#include <cstdint>
#include "types.hpp"

double cpu_monte_carlo_call_price(const OptionParams& p, int num_paths, std::uint64_t seed);
double cpu_monte_carlo_call_price_timed(const OptionParams& p,
                                        int num_paths,
                                        std::uint64_t seed,
                                        double* loop_ms_out);
