#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stddef.h>
#include <cstring>
#include <vector>

#include "printimage.hpp"

std::complex<double> function(std::complex<double>& z) {
  return z * z - std::complex<double>(-0.6, 0.6);
}

void julia_fatou() {
  const double step = 0.002;
  const std::complex<double> start(-2, -2);
  const size_t max_iter   = 70;
  const double norm_limit = 4;

  const size_t width = std::abs(double(start.real() * 2 / step));
  const size_t half_height =
    std::abs(double(start.imag() / step));  // calculate just one half because of rotation symmetrie
  std::vector<int> pixels(width * half_height);

  for (size_t i = 0; i < half_height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      std::complex<double> z(start.real() + step * j, start.real() + step * i);
      size_t counter = 0;
      do {
        z = function(z);
        ++counter;
      } while (std::abs(z) < norm_limit && counter <= max_iter);
      pixels.at(j + i * width) = counter;
    }
  }

  std::ofstream myfile("image.txt", std::ios::binary);
  myfile << half_height << "\n" << width << "\n";
  for (auto& pixel : pixels) {
    myfile << pixel << "\n";
  }
}

int main(int argc, char** argv) {
  if (std::strcmp(argv[1], "julia") == 0) {
    julia_fatou();
  }
  else if (std::strcmp(argv[1], "print") == 0) {
    printimage();
  }

  std::cout << "Successfully done!" << std::endl;
  return 0;
}
