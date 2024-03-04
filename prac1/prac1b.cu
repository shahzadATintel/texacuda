//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

//
// kernel routine
//

__global__ void my_first_kernel(float *x, float *y, float *res)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  res[tid] = x[tid] + y[tid];
}

//
// main code
//

int main(int argc, const char **argv)
{
  float *h_x, *d_x;
  int nblocks, nthreads, nsize, n;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks = 2;
  nthreads = 8;
  nsize = nblocks * nthreads;

  // allocate memory for array

  h_x = (float *)malloc(nsize * sizeof(float));
  h_y = (float *)malloc(nsize * sizeof(float));
  h_res = (float *)malloc(nsize * sizeof(float));

  checkCudaErrors(cudaMalloc((void **)&d_y, nsize * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_res, nsize * sizeof(float)));

  for (int i = 0; i < nsize; i++)
  {
    h_x[i] = h_y[i]= i;
  }


  // copy inputs to GPU device memory
  checkCudaErrors(cudaMemcpy(d_x, h_x, nsize * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, nsize * sizeof(float), cudaMemcpyHostToDevice));

  // execute kernel


  my_first_kernel<<<nblocks, nthreads>>>(d_x, d_x, d_res);

  getLastCudaError("my_first_kernel execution failed\n");
  checkCudaErrors(cudaMemcpy(h_y, d_y, nsize * sizeof(float), cudaMemcpyDeviceToHost));



  for (n = 0; n < nsize; n++)
  {
    printf(" n,  x  =  %d  %f \n", n, h_x[n]);
    if(h_res[n] != (n+n))
      printf("Results mismatch");
  }
  // free memory

  checkCudaErrors(cudaFree(d_x));
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
