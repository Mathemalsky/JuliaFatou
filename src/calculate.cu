#include "calculate.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <stdint.h>

#include <cuda.h>

#include "constants.hpp"
#include "variables.hpp"

const size_t BLOCKSIZE = 256;
// const double START_RE   = -1.75f;
// const double START_IM   = -1.75f;
const double NORM_LIMIT = 1000;

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

/*
__global__ static void calculatePixelsGPU(
  __int16_t* pixels, const size_t imageSize, const size_t width, const double step,
  const size_t max_iter) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < imageSize) {
    int idx             = id % width;
    int idy             = id / width;
    __int16_t iteration = 0;
    Complex z           = Complex(START_RE + idx * step, START_IM + idy * step);
    do {
      z = function(z);
      ++iteration;
    } while (iteration < max_iter && z.squaredAbs() < NORM_LIMIT);
    pixels[width * idy + idx] = iteration;
  }
}
*/

/*
// calculate just half of the pixels due to symmetrie
void julia_fatouCUDA(const char* filename, const double step, const size_t max_iter) {
  const size_t width       = std::abs(double(START_RE * 2 / step));
  const size_t half_height = std::abs(double(START_IM / step));
  const size_t imageSize   = half_height * width;

  __int16_t* pixels;
  cudaHostAlloc((void**) &pixels, imageSize * sizeof(__int16_t), 0);

  dim3 blockDim(BLOCKSIZE);
  dim3 gridDim(std::ceil(imageSize / (float) BLOCKSIZE));

  __int16_t* cudaPixels;
  cudaMalloc((void**) &cudaPixels, imageSize * sizeof(__int16_t));

  calculatePixelsGPU<<<gridDim, blockDim>>>(cudaPixels, imageSize, width, step, max_iter);

  cudaMemcpy(pixels, cudaPixels, imageSize * sizeof(__int16_t), cudaMemcpyDeviceToHost);

  cudaFree(cudaPixels);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  std::ofstream myfile(filename, std::ios::binary);
  myfile.write((char*) &width, sizeof(width));
  myfile.write((char*) &half_height, sizeof(half_height));
  myfile.write((char*) pixels, imageSize * sizeof(__int16_t));
  assert(myfile.fail() == 0 && "Could not write correctly!");
  myfile.close();

  cudaFreeHost(pixels);
}
*/

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
    } while (iteration < max_iter && z.squaredAbs() < NORM_LIMIT);

    // calculate colors and write
    cudaPixels[3 * (width * idy + idx)]     = std::round(red * iteration);
    cudaPixels[3 * (width * idy + idx) + 1] = std::round(green * iteration);
    cudaPixels[3 * (width * idy + idx) + 2] = std::round(blue * iteration);
  }
}

using namespace functionParameters;
// do calculation adjusted to the the displaying
void juliaFatouCUDA(Byte* textureImg) {
  const unsigned int imageSize = mainWindow::WIDTH * mainWindow::HEIGHT;

  // set up grid
  dim3 blockDim(BLOCKSIZE);
  dim3 gridDim(std::ceil(imageSize / (float) BLOCKSIZE));

  // allocate GPU memory
  Byte* cudaPixels;
  cudaMalloc((void**) &cudaPixels, imageSize * universal::RGB_COLORS);

  // do computation
  calculatePixelsGPU<<<gridDim, blockDim>>>(
    cudaPixels, imageSize, mainWindow::WIDTH, STEP, MAX_ITER, RE_START, IM_START, RED, GREEN, BLUE);

  // copy memory from GRAM to RAM
  cudaMemcpy(textureImg, cudaPixels, imageSize * universal::RGB_COLORS, cudaMemcpyDeviceToHost);

  // free the allocated memory on GPU
  cudaFree(cudaPixels);

  // print status cuda message
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  /*
  // DEBUG
  for(unsigned int i =0; i<imageSize * universal::RGB_COLORS; ++i) {
    textureImg[i] = 0.3;
  }
  */
}
