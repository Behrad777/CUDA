#include <iostream>


__global__
void vectorAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<n) C[i] = A[i]+B[i];

} //c=a+b size n 


//host code will call the device code 
void vectorAdd(float* A_h, float* B_h, float* C_h, int n){

    //first copy the host vectors into device pointer
    int size{n*sizeof(float)};
    float *A_d, *B_d, *C_d;


    //since cuda malloc excpects generic pointers, cast each
    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);
    //ommit error checking here, but just like regular malloc we may not find a contigous memory big enough 


    //source, dest, size, type of memcpy flag
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice);

    //addition kernel invocation code
    //several blocks of 256 threads until we reach # of blocks to reach n 

    //the call is <<< number of blocks, threads in a block>>>(parameters from the kernel function)
    vectorAddKernel<<< ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);


    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    //free allocated gpu space

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

int main(){

    float a[] ={1,2,3,4,5,6};
    float b[] ={1,2,3,4,5,6};
    float c[6]={};

    vectorAdd(a,b,c,6);

    for(const auto& item : c){
        std::cout<< item << "\n";

    }
}