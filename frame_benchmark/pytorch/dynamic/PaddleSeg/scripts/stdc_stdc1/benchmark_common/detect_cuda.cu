#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"

int main() {
  int version = -1;
  cudaError_t err = cudaRuntimeGetVersion(&version); 
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
    return -1;
  }
  int major = version / 1000;
  int minor = (version % 100) / 10; 
  printf("%d.%d\n", major, minor);
  return 0;
}
