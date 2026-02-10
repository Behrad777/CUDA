#include "rng.hpp"

__global__ void init_rng_kernel(curandState* states, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

curandState* rng_alloc_states(int num_threads){
    curandState* d_states = nullptr;
    cudaMalloc(&d_states, num_threads * sizeof(curandState));
    return d_states;
}


void rng_init(curandState* d_states, int num_states, std::uint64_t seed){
    int threads = THREADS_PER_BLOCK;
    int blocks  = (num_states + threads - 1) / threads;
    init_rng_kernel<<<blocks, threads>>>(d_states, seed);
}

void rng_free_states(curandState* d_states){
    cudaFree(d_states);
}

