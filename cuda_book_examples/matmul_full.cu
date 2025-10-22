#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_SIZE 16

/**
 * Tiled matrix multiplication kernel using shared memory
 * Computes C = A * B where:
 * A is M x K
 * B is K x N
 * C is M x N
 */
__global__ void matrixMultiplyTiled(float* A, float* B, float* C,
                                     int M, int K, int N) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index of C element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tile from A into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Naive matrix multiplication kernel (for comparison)
 */
__global__ void matrixMultiplyNaive(float* A, float* B, float* C,
                                     int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Initialize matrix with random values
void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

// CPU matrix multiplication for verification
void matrixMultiplyCPU(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify GPU results against CPU
bool verifyResults(float* gpuResult, float* cpuResult, int size, float epsilon = 0.01) {
    for (int i = 0; i < size; i++) {
        if (fabs(gpuResult[i] - cpuResult[i]) > epsilon) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f, DIFF: %f\n",
                   i, gpuResult[i], cpuResult[i], fabs(gpuResult[i] - cpuResult[i]));
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Matrix dimensions (M x K) * (K x N) = (M x N)
    int M = 1024;  // Rows of A and C
    int K = 512;   // Cols of A, rows of B
    int N = 1024;  // Cols of B and C

    if (argc >= 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }

    printf("Matrix Multiplication: (%d x %d) * (%d x %d) = (%d x %d)\n",
           M, K, K, N, M, N);
    printf("Using tile size: %d x %d\n\n", TILE_SIZE, TILE_SIZE);

    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_tiled = (float*)malloc(sizeC);
    float* h_C_naive = (float*)malloc(sizeC);
    float* h_C_cpu = (float*)malloc(sizeC);

    // Initialize matrices
    srand(42);
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block dimensions: (%d, %d)\n\n", blockDim.x, blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ========== Tiled Matrix Multiplication ==========
    printf("Running tiled matrix multiplication...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixMultiplyTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float tiledTime;
    CUDA_CHECK(cudaEventElapsedTime(&tiledTime, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, sizeC, cudaMemcpyDeviceToHost));
    printf("Tiled kernel time: %.3f ms\n", tiledTime);

    float tiledGFlops = (2.0f * M * N * K * 1e-9) / (tiledTime * 1e-3);
    printf("Tiled performance: %.2f GFLOPS\n\n", tiledGFlops);

    // ========== Naive Matrix Multiplication ==========
    printf("Running naive matrix multiplication...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixMultiplyNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naiveTime;
    CUDA_CHECK(cudaEventElapsedTime(&naiveTime, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, sizeC, cudaMemcpyDeviceToHost));
    printf("Naive kernel time: %.3f ms\n", naiveTime);

    float naiveGFlops = (2.0f * M * N * K * 1e-9) / (naiveTime * 1e-3);
    printf("Naive performance: %.2f GFLOPS\n\n", naiveGFlops);

    printf("Speedup (Tiled vs Naive): %.2fx\n\n", naiveTime / tiledTime);

    // ========== CPU Verification ==========
    printf("Running CPU verification...\n");
    clock_t cpuStart = clock();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, M, K, N);
    clock_t cpuEnd = clock();
    double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU time: %.3f ms\n\n", cpuTime);

    // Verify results
    printf("Verifying tiled kernel results... ");
    if (verifyResults(h_C_tiled, h_C_cpu, M * N)) {
        printf("PASSED!\n");
    } else {
        printf("FAILED!\n");
    }

    printf("Verifying naive kernel results... ");
    if (verifyResults(h_C_naive, h_C_cpu, M * N)) {
        printf("PASSED!\n");
    } else {
        printf("FAILED!\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_tiled);
    free(h_C_naive);
    free(h_C_cpu);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
