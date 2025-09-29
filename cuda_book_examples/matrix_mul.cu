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


__global__ void matrixMultiplicationKernel(float * M, float * N, float * P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < width) && (col < width)) {
        float pVal = 0;
        for(int k=0; k<width; k++){
            pVal += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = pVal;
    }
}