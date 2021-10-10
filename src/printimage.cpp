#include "printimage.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#include <pngwriter.h>

// read the input data
__int16_t* readimage(size_t& half_height, size_t& width, const char* filename) {
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
  const double blue) {
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

  for (size_t i = 0; i < half_size; ++i) {
    double intensity = double(pixels[i]) / maxiter;

    const double pixel_red   = std::round(red * intensity);    // red
    const double pixel_blue  = std::round(blue * intensity);   // blue
    const double pixel_green = std::round(green * intensity);  // green

    unsigned int x = i % width;
    unsigned int y = i / width;
    png.plot(x, y, pixel_red, pixel_green, pixel_blue);
    png.plot(width - 1 - x, height - 1 - y, pixel_red, pixel_green, pixel_blue);
  }
  png.close();

  free(pixels);
  std::cout << "maximum Iterations: " << maxiter << std::endl;
}
