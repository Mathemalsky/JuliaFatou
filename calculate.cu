#include "calculate.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <stdint.h>

#include <cuda.h>

struct Complex {
  double p_re, p_im;
  __device__ Complex(const double re, const double im) : p_re(re), p_im(im) {}
  /*
  __device__ double re() const {
    return this->p_re;
  }
  _device__ double& re() {
    return this->p_re;
  }
  _device__ double im() const {
    return this->p_im;
  }
  _device__ double& im() {
    return this->p_im;
  }
  */
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

/*
for (size_t i = 0; i < half_height; ++i) {
  for (size_t j = 0; j < width; ++j) {
    std::complex<double> z(start.real() + step * j, start.real() + step * i);
    __int16_t counter = 0;
    do {
      z = function(z);
      ++counter;
    } while (std::abs(z) < norm_limit && counter < (__int16_t) max_iter);
    pixels[j + i * width] = counter;
  }
}
*/



__device__ static Complex function(const Complex& z) {
  return z * z + Complex(0.2, -0.6);
}

__global__
void calculatePixelsGPU(__int16_t* pixels, const double step, const size_t max_iter) {
  int i = blockIdx.x;
  int j = blockIdx.y;
  size_t iteration = 0;
  Complex z = Complex(-1.75f + i*step, -1.75f + j*step);
  do {
    z = function(z);
  } while(iteration < max_iter && z.squaredAbs() <4);
  pixels[gridDim.x * j + i] = iteration;
}

// calculate just half of the pixels due to symmetrie
void julia_fatouCUDA(const char* filename, const double step, const size_t max_iter) {
  const double start_re = -1.75f;
  const double start_im = -1.75f;
  //const double norm_limit = 4;

  const size_t width       = std::abs(double(start_re * 2 / step));
  const size_t half_height = std::abs(double(start_im / step));
  const size_t imageSize   = half_height * width;
  __int16_t* pixels        = (__int16_t*) malloc(imageSize * sizeof(__uint16_t));

  dim3 grid(width,half_height);

  __int16_t* cudaPixels;
  cudaMalloc((void **)&cudaPixels, imageSize*sizeof(__int16_t));

  /*
  size_t* cudaImageSize;
  cudaMalloc((void **)&cudaImageSize, sizeof (size_t));
  cudaMemcpy(cudaImageSize, imageSize, sizeof (size_t), cudaMemcpyHostToDevice);

  size_t* cudaHalfHeight;
  cudaMalloc((void **)&cudaHalfHeight, sizeof (size_t));
  cudaMemcpy(cudaHalfHeight, half_height, sizeof (size_t), cudaMemcpyHostToDevice);

  double* cudaStep;
  cudaMalloc((void **)&cudaStep, sizeof (double));
  cudaMemcpy(cudaStep, step, sizeof (double), cudaMemcpyHostToDevice);

  size_t* cudaMaxIter;
  cudaMalloc((void **)&cudaMaxIter, sizeof (size_t));
  cudaMemcpy(cudaMaxIter, max_iter, sizeof (size_t), cudaMemcpyHostToDevice);
  */

  calculatePixelsGPU<<<grid, 1>>>(cudaPixels, step, max_iter);

  cudaMemcpy(pixels, cudaPixels, imageSize * sizeof (__int16_t), cudaMemcpyDeviceToHost);

  cudaFree(cudaPixels);
  /*
  cudaFree(cudaImageSize);
  cudaFree(cudaHalfHeight);
  cudaFree(cudaStep);
  cudaFree(cudaMaxIter);
  */

  printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );

  std::ofstream myfile(filename, std::ios::binary);
  myfile.write((char*) &width, sizeof(width));
  myfile.write((char*) &half_height, sizeof(half_height));
  myfile.write((char*) pixels, imageSize * sizeof(__int16_t));
  assert(myfile.fail() == 0 && "Could not write correctly!");
  myfile.close();

  free(pixels);
}
