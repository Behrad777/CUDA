#pragma once
#include "config.hpp"
#include "types.hpp"

//helper for copying the parameters into device memory
void upload_option_params(const OptionParams& params);

//helper for launching the kernel
void launch_monte_carlo(float* d_payoffs, curandState* d_states, int num_paths);
