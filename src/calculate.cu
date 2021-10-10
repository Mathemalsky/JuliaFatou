#include "calculate.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <stdint.h>

#include <cuda.h>

#include <iostream>

struct Complex {
  double p_re, p_im;
  __device__ Complex(const double re, const double im) : p_re(re), p_im(im) {
  }
  __device__ Complex operator*(const Complex& a) const {
    return Complex(p_re * a.p_re - p_im * a.p_im, p_re * a.p_im + p_im * a.p_re);
  }
  __device__ Complex operator+(const Complex& a) const {
    return Complex(p_re + a.p_re, p_im + a.p_im);
  }
  __device__ double squaredAbs() const {
    return p_re * p_re + p_im * p_im;
  }
};

__device__ static Complex function(const Complex& z) {
  return z * z + Complex(0.2, -0.6);
}

__global__ void calculatePixelsGPU(__int16_t* pixels, const double step, const size_t max_iter) {
  int i            = blockIdx.x;
  int j            = blockIdx.y;
  size_t iteration = 0;
  Complex z        = Complex(-1.75f + i * step, -1.75f + j * step);
  do {
    z = function(z);
    ++iteration;
  } while (iteration < max_iter && z.squaredAbs() < 4);
  pixels[gridDim.x * j + i] = iteration;
}

// calculate just half of the pixels due to symmetrie
void julia_fatouCUDA(const char* filename, const double step, const size_t max_iter) {
  const double start_re = -1.75f;
  const double start_im = -1.75f;
  // const double norm_limit = 4;

  const size_t width       = std::abs(double(start_re * 2 / step));
  const size_t half_height = std::abs(double(start_im / step));
  const size_t imageSize   = half_height * width;
  //__int16_t* pixels        = (__int16_t*) malloc(imageSize * sizeof(__int16_t));
  __int16_t* pixels;
  cudaHostAlloc((void**) &pixels, imageSize * sizeof(__int16_t), 0);

  dim3 grid(width, half_height);

  __int16_t* cudaPixels;
  cudaMalloc((void**) &cudaPixels, imageSize * sizeof(__int16_t));

  calculatePixelsGPU<<<grid, 1>>>(cudaPixels, step, max_iter);

  cudaMemcpy(pixels, cudaPixels, imageSize * sizeof(__int16_t), cudaMemcpyDeviceToHost);

  cudaFree(cudaPixels);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  std::ofstream myfile(filename, std::ios::binary);
  myfile.write((char*) &width, sizeof(width));
  myfile.write((char*) &half_height, sizeof(half_height));
  myfile.write((char*) pixels, imageSize * sizeof(__int16_t));
  assert(myfile.fail() == 0 && "Could not write correctly!");
  myfile.close();

  // free(pixels);
  cudaFreeHost(pixels);
}
