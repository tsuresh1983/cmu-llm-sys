#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace std;
using std::generate;
using std::vector;

__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N){
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int N) {
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    vecAddKernel<<<num_blocks, threads_per_block>>>(A_d, B_d, C_d, N);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

void print_result(vector<float> &a, vector<float> &b, vector<float> &c, int N) {
  // For every element...
  for (int i = 0; i < N; i++) {
    printf("%d::: %f  +  %f = %f\n", i, a[i], b[i], c[i]);
  }
}


int main() {

  int n = 1024;
  int iterations = 1000000;
  for(int i=0; i<iterations;i++){
    // Host vectors
    vector<float> h_a(n);
    vector<float> h_b(n);
    vector<float> h_c(n);
    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    vecAdd(h_a.data(), h_b.data(), h_c.data(), n);
    print_result(h_a, h_b, h_c, n);
  }
}

