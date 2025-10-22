#include <cuda_runtime.h>
  #include <iostream>
  #include <cassert>
  #include <vector>

  using namespace std;

  #define TILE_WIDTH 16  // Must be compile-time constant for shared memory

  __global__ void matrixMulTilesKernel(float* M, float* N, float* P, int Width) {
      __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
      __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

      int bx = blockIdx.x;
      int by = blockIdx.y;
      int tx = threadIdx.x;
      int ty = threadIdx.y;

      // Identify the row and col of the P element to work on
      int Row = by * TILE_WIDTH + ty;
      int Col = bx * TILE_WIDTH + tx;

      float Pvalue = 0;

      // Loop over the M and N tiles required to compute P element
      int numTiles = (Width + TILE_WIDTH - 1) / TILE_WIDTH;  // Ceiling division

      for(int ph = 0; ph < numTiles; ++ph) {
          // Collaborative loading of M and N tiles into shared memory with bounds checking
          int mCol = ph * TILE_WIDTH + tx;
          int nRow = ph * TILE_WIDTH + ty;

          // Load M tile with boundary check
          if (Row < Width && mCol < Width) {
              Mds[ty][tx] = M[Row * Width + mCol];
          } else {
              Mds[ty][tx] = 0.0f;
          }

          // Load N tile with boundary check
          if (nRow < Width && Col < Width) {
              Nds[ty][tx] = N[nRow * Width + Col];
          } else {
              Nds[ty][tx] = 0.0f;
          }

          __syncthreads();

          // Compute partial dot product
          for(int k = 0; k < TILE_WIDTH; ++k) {
              Pvalue += Mds[ty][k] * Nds[k][tx];
          }
          __syncthreads();
      }

      // Write result with boundary check
      if (Row < Width && Col < Width) {
          P[Row * Width + Col] = Pvalue;
      }
  }
