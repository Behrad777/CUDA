#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include "cuda/kernels.cuh"

//geometric brownian motion model parameters, constants in device memory
__constant__ float d_S0;
__constant__ float d_K;
__constant__ float d_r;
__constant__ float d_sigma;
__constant__ float d_T;
const size_t size{sizeof(float)};




// For each path, generate random # from initial seed, compute possible stock price at maturity using
// GBM model
__global__ void monte_carlo_kernel(float* payoffs, curandState* states, int num_paths, int num_states){
    int thread_id = blockDim.x*blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // stride size for the threads

    //load RNG state into a register (faster than using global memory every draw)
    curandState local = states[thread_id];

    //responsible for exponential growth of decay of price over time
    float drift = (d_r - 0.5f * d_sigma * d_sigma) * d_T;

    //responsible for random fluctuations in price 
    float volatility   = d_sigma * sqrtf(d_T);


    for (int path = thread_id; path < num_paths; path += stride) {
        float z  = curand_normal(&local); // takes the current state, returns a random standard normal distributed number on 0, then updates the state for the next stride 
        float ST = d_S0 * expf(drift + volatility * z);
        payoffs[path] = fmaxf(ST - d_K, 0.0f); //difference from strike price
    }

    //write back rng to maintain randomness across launches 
    states[thread_id] = local;
}

void upload_option_params(const OptionParams& params){
    cudaMemcpyToSymbol(d_S0, &params.S0, size);
    cudaMemcpyToSymbol(d_K, &params.K, size);
    cudaMemcpyToSymbol(d_r, &params.r, size);
    cudaMemcpyToSymbol(d_sigma, &params.sigma, size);
    cudaMemcpyToSymbol(d_T, &params.T, size);
}

void launch_monte_carlo(float* d_payoffs, curandState* d_states, int num_paths){
    int blocks = (num_paths + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; 
    monte_carlo_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_payoffs, d_states, num_paths, blocks*THREADS_PER_BLOCK);
}
