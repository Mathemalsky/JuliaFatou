#include "printimage.hpp"

#include <cmath>

#include "constants.hpp"

#include <pngwriter.h>

void printImage(const char* filename, Byte* pixels, const unsigned int width, const unsigned int height) {
  pngwriter png(width, height, 0, filename);
  png.setcompressionlevel(9);
  const unsigned int size = width * height;
  for (unsigned int i = 0; i < size; ++i) {
    const unsigned int x    = i % width + 1;
    const unsigned int y    = i / width + 1;
    const double pixelRed   = (double) pixels[universal::RGB_COLORS * i] / universal::MAX_BYTE;
    const double pixelGreen = (double) pixels[universal::RGB_COLORS * i + 1] / universal::MAX_BYTE;
    const double pixelBlue  = (double) pixels[universal::RGB_COLORS * i + 2] / universal::MAX_BYTE;
    png.plot(x, y, pixelRed, pixelGreen, pixelBlue);
  }
  png.close();
}
