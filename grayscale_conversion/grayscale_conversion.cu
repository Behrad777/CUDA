//---CUDA kernel that converts linearized arrays of RGB images into grayscale
//each pixel is encoded as 3 consecutive characters for 3 channels 
/*
A linearized version of this type of image would look like RGB RGB RGB 

*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void colorToGrayKernel(unsigned char* P_out, unsigned char* P_in, int width, int height){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    //prune extra threads
    if(col < width && row <height){
        int grey_offset = row*width + col;

        int rgb_offset = grey_offset*3; //3 channels 

        //get the 3 consecutive channels 
        unsigned char r = P_in[rgb_offset];
        unsigned char g = P_in[rgb_offset+1];
        unsigned char b = P_in[rgb_offset+2];


        //use common rescaling function to store the new single value
        P_out[grey_offset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }

}
static void ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s input.png output.png [iters]\n", argv[0]);
        return 1;
    }
    int iters = (argc >= 4) ? std::atoi(argv[3]) : 1000;
    if (iters <= 0) iters = 1000;

    int w=0, h=0, n=0;
    unsigned char* h_rgb = stbi_load(argv[1], &w, &h, &n, 3);
    if (!h_rgb) {
        std::fprintf(stderr, "stbi_load failed: %s\n", stbi_failure_reason());
        return 2;
    }

    size_t pixels     = (size_t)w * h;
    size_t rgb_bytes  = pixels * 3;
    size_t gray_bytes = pixels;

    unsigned char *d_rgb=nullptr, *d_gray=nullptr;
    ck(cudaMalloc(&d_rgb,  rgb_bytes),  "cudaMalloc d_rgb");
    ck(cudaMalloc(&d_gray, gray_bytes), "cudaMalloc d_gray");
    ck(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice), "H2D rgb");

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // Warmup
    for (int i = 0; i < 10; i++) colorToGrayKernel<<<grid, block>>>(d_gray, d_rgb, w, h);
    ck(cudaDeviceSynchronize(), "warmup sync");

    // CUDA event timing (kernel only)
    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start), "event create start");
    ck(cudaEventCreate(&stop),  "event create stop");

    ck(cudaEventRecord(start), "event record start");
    for (int i = 0; i < iters; i++) {
        colorToGrayKernel<<<grid, block>>>(d_gray, d_rgb, w, h);
    }
    ck(cudaEventRecord(stop), "event record stop");
    ck(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

    double ms_per_iter = ms / iters;

    // Approx bytes moved by kernel: read 3B, write 1B per pixel (ignores caches, overhead)
    double bytes_per_iter = (double)pixels * 4.0;
    double gbps = (bytes_per_iter / (ms_per_iter / 1000.0)) / 1e9;

    std::printf("Image: %dx%d (%zu pixels)\n", w, h, pixels);
    std::printf("Iters: %d\n", iters);
    std::printf("Kernel: %.6f ms/iter\n", ms_per_iter);
    std::printf("Est. throughput: %.2f GB/s (3B read + 1B write per pixel)\n", gbps);

    // Copy back once and write output
    unsigned char* h_gray = (unsigned char*)std::malloc(gray_bytes);
    ck(cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost), "D2H gray");
    if (!stbi_write_png(argv[2], w, h, 1, h_gray, w)) {
        std::fprintf(stderr, "stbi_write_png failed\n");
        return 4;
    }

    std::free(h_gray);
    stbi_image_free(h_rgb);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_gray);
    cudaFree(d_rgb);
    return 0;
}