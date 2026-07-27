#include <stdio.h>
#include <algorithm>

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int *in, int *out) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  // read input data into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  __syncthreads();

  // apply stencil
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
    result += temp[lindex + offset];
  }

  out[gindex] = result;
}

void fill_ints(int *x, int n) {
  std::fill_n(x, n, 1);
}

int main(void) {
  int *h_in, *h_out;
  int *d_in, *d_out;

  int size = (N + 2 * RADIUS) * sizeof(int);
  h_in = (int *)malloc(size);
  h_out = (int *)malloc(size);
  fill_ints(h_in, N + 2 * RADIUS);
  fill_ints(h_out, N + 2 * RADIUS);

  // allocate memory for device
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // H2D
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out, size, cudaMemcpyHostToDevice);

  stencil_1d<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);

  // D2H
  cudaMemcpy(h_out , d_out, size, cudaMemcpyDeviceToHost);

  // error checking
  for (int i = 0; i < N + 2 * RADIUS; ++i) {
    if (i < RADIUS || i >= N + RADIUS) {
      if (h_out[i] != 1) {
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, h_out[i], 1);
      }
    }
    else {
      if (h_out[i] != 1 + 2 * RADIUS) {
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, h_out[i], 1 + 2*RADIUS);
      }
    }
  }

  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);

  printf("Success!\n");
  return 0;
}
