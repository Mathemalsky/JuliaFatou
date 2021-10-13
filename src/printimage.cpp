#include "printimage.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#include <pngwriter.h>

static double modifier(const double x) {
  return 0.5f + 4 * pow(x - 0.5f, 3);
}

// read the input data
static __int16_t* readimage(size_t& half_height, size_t& width, const char* filename) {
  std::ifstream myfile(filename, std::ios::binary);

  myfile.read((char*) &width, sizeof(width));
  myfile.read((char*) &half_height, sizeof(half_height));
  __int16_t* pixels = (__int16_t*) malloc(half_height * width * sizeof(__int16_t));
  myfile.read((char*) pixels, half_height * width * sizeof(__int16_t));

  assert(myfile.fail() == 0 && "Couldn't read file correctly.");
  myfile.close();
  return pixels;
}

void printimage(
  const char* inputFilename, const char* outputFilename, const double red, const double green,
  const double blue, const double red2, const double green2, const double blue2) {
  size_t half_height, width;
  int16_t* pixels = readimage(half_height, width, inputFilename);

  const size_t half_size = half_height * width;
  const size_t height    = 2 * half_height;
  int maxiter            = 0;
  for (size_t i = 0; i < half_size; ++i) {
    if (pixels[i] > maxiter) {
      maxiter = pixels[i];
    }
  }

  pngwriter png(width, height, 0, outputFilename);
  png.setcompressionlevel(9);

  for (size_t i = 0; i < half_size; ++i) {
    const double intensity   = modifier(double(pixels[i]) / maxiter);
    const double pixel_red   = red * intensity + red2 * (1 - intensity);      // red
    const double pixel_green = green * intensity + green2 * (1 - intensity);  // green
    const double pixel_blue  = blue * intensity + blue2 * (1 - intensity);    // blue
    unsigned int x           = i % width + 1;
    unsigned int y           = i / width + 1;
    png.plot(x, y, pixel_red, pixel_green, pixel_blue);
    png.plot(width - x, height - y, pixel_red, pixel_green, pixel_blue);
  }
  png.close();

  free(pixels);
  std::cout << "maximum Iterations: " << maxiter << std::endl;
}
