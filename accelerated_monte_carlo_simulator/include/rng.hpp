#pragma once

#include <curand_kernel.h>
#include <cstdint>
#include "config.hpp"

curandState* rng_alloc_states(int num_threads);
void rng_init(curandState* d_states, int num_threads, std::uint64_t seed);
void rng_free_states(curandState* d_states);
