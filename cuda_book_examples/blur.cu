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


int BLUR_SIZE = 3;

__global__ void blurKernel(unsigned char * in, unsigned char * out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < w && col < h){
        int pixVal = 0;
        int pixels = 0;

        //get average of the surroundings of BLUR_SIZE * BLUR_SIZE
        for(int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
            for(int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // verify if we have valid pixel
                if(curRow > 0 && curRow < h && curCol > 0 && curCol < w){
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }

        // write our new pixel value out
        out[row * w + col ] = (unsigned char)(pixVal/pixels);
    }
}