#include "printimage.hpp"

#include <cmath>
#include <string>

#include "calculate.hpp"
#include "constants.hpp"
#include "variables.hpp"

#include <pngwriter.h>

static inline Byte doubleToByte(const double a) {
  return std::round(a * universal::MAX_BYTE);
}

static inline std::string byteToHexstring(const Byte byte) {
  const std::string table = "0123456789abcdef";
  std::string hex;
  hex.reserve(2);
  hex.push_back(table[byte / 16]);
  hex.push_back(table[byte % 16]);
  return hex;
}

static inline std::string toHexstring(const double a) {
  return byteToHexstring(doubleToByte(a));
}

void saveImage(const char* filename, Byte* pixels, const unsigned int width, const unsigned int height) {
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

using namespace functionParameters;

void screenshot() {
  std::string filename = std::to_string(functionParameters::SCREENSHOT_WIDTH) + "x";
  filename += std::to_string(functionParameters::SCREENSHOT_HEIGHT) + "_";
  filename += std::to_string(RE_START) + "_" + std::to_string(IM_START) + "_" + std::to_string(STEP) + "_";
  filename += toHexstring(D_RED) + toHexstring(D_GREEN) + toHexstring(D_BLUE) + ".png";

  // Error here calloc has to be changed to malloc.
  // call to screenshot seems to have no effect on the data
  Byte* pixels = (Byte*) std::calloc(
    universal::RGB_COLORS * functionParameters::SCREENSHOT_WIDTH * functionParameters::SCREENSHOT_HEIGHT, 1);
  singleBigFrame(pixels);

  saveImage(filename.c_str(), pixels, functionParameters::SCREENSHOT_WIDTH, functionParameters::SCREENSHOT_HEIGHT);
  std::cout << "Current image has been saved as <" + filename + ">\n";
  free(pixels);
}
