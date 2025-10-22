## How to run

### compile cuda using

e.g.
nvcc -O3 -arch=sm_60 -o matmul_full  matmul_full.cu -Xcompiler -fPIC

### Run
./matmul_full 256 128 256