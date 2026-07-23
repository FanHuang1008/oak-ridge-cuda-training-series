#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 4096;
const int block_size = 256;  // CUDA maximum is 1024

// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ds) {
    C[idx] = A[idx] + B[idx];
  }
}

int main(){
  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // allocate space for vectors in host memory
  h_A = new float[DSIZE];
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];

  // initialize vectors in host memory
  for (int i = 0; i < DSIZE; ++i) {
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;
  }

  // allocate space for vectors in device memory
  cudaMalloc(&d_A, DSIZE * sizeof(float));
  cudaMalloc(&d_B, DSIZE * sizeof(float));
  cudaMalloc(&d_C, DSIZE * sizeof(float));

  cudaCheckErrors("cudaMalloc failure");

  // copy vector A and B to device
  cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

  cudaCheckErrors("cudaMemcpy H2D failure");

  vadd<<<(DSIZE + (block_size - 1)) / block_size, block_size>>>(d_A, d_B, d_C, DSIZE);

  cudaCheckErrors("kernel launch failure");

  // copy vector C from device to host
  cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);
  return 0;
}
  
