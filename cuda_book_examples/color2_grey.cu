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


int CHANNELS = 255;


__global__ void colortoGrayscaleConversion(unsigned char * Pout, unsigned char * Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < width && row < height){
        // get 1D offset for the grayscale image
        int grayOffset = row * width + col;
        
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}