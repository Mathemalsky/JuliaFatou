#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "printimage.hpp"

std::complex<double> function(std::complex<double>& z) {
  return z * z - std::complex<double>(-0.6, 0.6);
}

// calculate just half of the pixels due to symmetrie
void julia_fatou(const char* filename) {
  const double step = 0.005;
  const std::complex<double> start(-2, -2);
  const size_t max_iter   = 70;
  const double norm_limit = 4;

  const size_t width = std::abs(double(start.real() * 2 / step));
  const size_t half_height = std::abs(double(start.imag() / step));
  int16_t* pixels = (int16_t*)malloc(width * half_height);

  for (size_t i = 0; i < half_height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      std::complex<double> z(start.real() + step * j, start.real() + step * i);
      size_t counter = 0;
      do {
        z = function(z);
        ++counter;
      } while (std::abs(z) < norm_limit && counter <= max_iter);
      pixels[j + i * width] = counter;
    }
  }

  FILE* myfile = fopen(filename,"wb");
  // write half_height, width
  fwrite(pixels,sizeof(pixels[0]),half_height*width,myfile);
  fclose(myfile);
}

int main(int argc, char** argv) {
  if (std::strcmp(argv[1], "julia") == 0) {
    julia_fatou(argv[2]);
  }
  else if (std::strcmp(argv[1], "print") == 0) {
    printimage(argv[2]);
  }

  std::cout << "Successfully done!" << std::endl;
  return 0;
}
