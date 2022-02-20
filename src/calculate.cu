#include "calculate.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <stdint.h>

#include <cuda.h>

#include "constants.hpp"
#include "variables.hpp"

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
  return z * z + Complex(-0.78, -0.18);
}

// iterate the function and calculate the color
__global__ static void calculatePixelsGPU(
  Byte* cudaPixels, const unsigned int imageSize, const unsigned int width, const float step,
  const unsigned int max_iter, const float startRe, const float startIm, const float red,
  const float green, const float blue) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < imageSize) {
    int idx             = id % width;
    int idy             = id / width;
    __int16_t iteration = 0;
    Complex z           = Complex(startRe + idx * step, startIm + idy * step);
    do {
      z = function(z);
      ++iteration;
    } while (iteration < max_iter && z.squaredAbs() < functionParameters::NORM_LIMIT);

    // calculate colors and write
    cudaPixels[3 * (width * idy + idx)]     = std::round((red * iteration * 255) / max_iter);
    cudaPixels[3 * (width * idy + idx) + 1] = std::round((green * iteration * 255) / max_iter);
    cudaPixels[3 * (width * idy + idx) + 2] = std::round((blue * iteration * 255) / max_iter);
  }
}

using namespace functionParameters;
// do calculation adjusted to the the displaying
void juliaFatouCUDA(Byte* textureImg, void* cudaPixels) {
  const unsigned int imageSize = mainWindow::WIDTH * mainWindow::HEIGHT;

  // set up grid
  dim3 blockDim(settingsGPU::BLOCKSIZE, 1, 1);
  dim3 gridDim(std::ceil(imageSize / (float) settingsGPU::BLOCKSIZE), 1, 1);

  // do computation
  calculatePixelsGPU<<<gridDim, blockDim>>>(
    (Byte*) cudaPixels, imageSize, mainWindow::WIDTH, STEP, MAX_ITER, RE_START, IM_START, RED,
    GREEN, BLUE);

  // copy memory from GRAM to RAM
  cudaMemcpy(textureImg, cudaPixels, imageSize * universal::RGB_COLORS, cudaMemcpyDeviceToHost);
}

void* allocateGraphicsMemory() {
  void* cudaPixels;
  const unsigned int imageSize = mainWindow::WIDTH * mainWindow::HEIGHT;
  cudaMalloc((void**) &cudaPixels, imageSize * universal::RGB_COLORS);
  return cudaPixels;
}

void freeGraphicsMemory(void* cudaPixels) {
  cudaFree(cudaPixels);
}
