#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstdint>

#include "config.hpp"          // NUM_PATHS, THREADS_PER_BLOCK (threads per block)
#include "types.hpp"           // OptionParams
#include "rng.hpp"             // rng_alloc_states, rng_init, rng_free_states
#include "cuda/kernels.cuh"    // upload_option_params, launch_monte_carlo

static void cuda_check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

int main()
{
    // 1) Set option parameters (host)
    

    //parameters are from apple stock feb 10 2026
    //simulate a call after 30 days I think its gonna raise to $290 per share
    // sigma: 30-day implied volatility for AAPL is around 23–24% (annualized).
    // r: // assume a 5% annual risk-free rate 
    OptionParams params{};
    params.S0 = 274.03f;
    params.K = 290.0f;
    params.r = 0.05f;
    params.sigma = 0.24f;
    params.T = 0.0822f; //expires after 30 days

    // 2) Upload params to device constant memory
    upload_option_params(params);

    // 3) Allocate device payoff array
    float* d_payoffs = nullptr;
    cuda_check(cudaMalloc(&d_payoffs, NUM_PATHS * sizeof(float)), "cudaMalloc(d_payoffs)");

    // 4) Compute the launch grid the same way your launch_monte_carlo does
    const int blocks = (NUM_PATHS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int num_states = blocks * THREADS_PER_BLOCK;

    // 5) Allocate + init RNG states (one per launched thread)
    curandState* d_states = rng_alloc_states(num_states);
    if (!d_states) {
        std::cerr << "rng_alloc_states returned nullptr\n";
        return 1;
    }

    const std::uint64_t seed = 123456789ULL;
    rng_init(d_states, num_states, seed);

    // Ensure RNG init completed successfully
    cuda_check(cudaGetLastError(), "init_rng_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after rng_init");

    // 6) Launch Monte Carlo kernel via wrapper
    launch_monte_carlo(d_payoffs, d_states, NUM_PATHS);

    cuda_check(cudaGetLastError(), "monte_carlo_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after monte_carlo");

    // 7) Copy payoffs back
    std::vector<float> h_payoffs(NUM_PATHS);
    cuda_check(cudaMemcpy(h_payoffs.data(),
                          d_payoffs,
                          NUM_PATHS * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H payoffs");



    
    // 8) Reduce on CPU: average payoff, then discount
    double sum = std::accumulate(h_payoffs.begin(), h_payoffs.end(), 0.0);
    double mean_payoff = sum / static_cast<double>(NUM_PATHS);
    double option_price = std::exp(-params.r * params.T) * mean_payoff;

    std::cout << "NUM_PATHS      : " << NUM_PATHS << "\n";
    std::cout << "threads/block  : " << THREADS_PER_BLOCK << "\n";
    std::cout << "blocks         : " << blocks << "\n";
    std::cout << "num_states     : " << num_states << "\n";
    std::cout << "Option price   : " << option_price << "\n";

    // 9) Cleanup
    rng_free_states(d_states);
    cuda_check(cudaFree(d_payoffs), "cudaFree(d_payoffs)");

    return 0;
}
