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

const size_t DSIZE = 16384;  // matrix side dimension
const int block_size = 256;  // CUDA maximum is 1024

// matrix row-sum kernel
__global__ void row_sums(const size_t ds, const float *A, float *sums) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ds) {
    float sum = 0.0f;
    for (int i = 0; i < ds; ++i) {
      sum += A[idx * ds + i];
    }
    sums[idx] = sum;
  }
}

// matrix column-sum kernel
__global__ void column_sums(const size_t ds, const float *A, float *sums) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ds) {
    float sum = 0.0f;
    for (int i = 0; i < ds; ++i) {
      sum += A[i * ds + idx];
    }
    sums[idx] = sum;
  }
}

bool validate(const size_t sz, const float *data) {
  for (size_t i = 0; i < sz; ++i) {
    if (data[i] != static_cast<float>(sz)) {
      printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], static_cast<float>(sz));
      return false;
    }

    return true;
  }
}

int main() {
  float * h_A, *h_sums, *d_A, *d_sums;

  h_A = new float[DSIZE * DSIZE];
  h_sums = new float[DSIZE]();

  for (int i = 0; i < DSIZE * DSIZE; ++i) {
    h_A[i] = 1.0f;
  }

  cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_sums, DSIZE * sizeof(float));

  cudaCheckErrors("cudaMalloc failure");

  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);

  cudaCheckErrors("cudaMemcpy H2D failure");

  row_sums<<<(DSIZE + block_size - 1)/block_size, block_size>>>(DSIZE, d_A, d_sums);

  cudaCheckErrors("kernel launch failure");

  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

  cudaCheckErrors("cudaMemcpy D2H failure");

  if (!validate(DSIZE, h_sums)) {
    return -1;
  }
  printf("row sums correct!\n");

  cudaMemset(d_sums, 0.0f, DSIZE * sizeof(float));

  column_sums<<<(DSIZE + block_size - 1)/block_size, block_size>>>(DSIZE, d_A, d_sums);

  cudaCheckErrors("kernel launch failure");

  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

  cudaCheckErrors("cudaMemcpy D2H failure");

  if (!validate(DSIZE, h_sums)) {
    return -1;
  }
  printf("column sums correct!\n");

  return 0;
}
  
