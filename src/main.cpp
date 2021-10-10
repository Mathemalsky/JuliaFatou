#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "calculate.hpp"
#include "printimage.hpp"

std::complex<double> function(const std::complex<double>& z) {
  return z * z - std::complex<double>(-0.2, 0.6);
}

// calculate just half of the pixels due to symmetrie
void julia_fatou(const char* filename, const double step = 0.005, const size_t max_iter = 255) {
  const std::complex<double> start(-1.75, -1.75);
  const double norm_limit = 4;

  const size_t width       = std::abs(double(start.real() * 2 / step));
  const size_t half_height = std::abs(double(start.imag() / step));
  __int16_t* pixels        = (__int16_t*) malloc(width * half_height * sizeof(__uint16_t));

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

  std::ofstream myfile(filename, std::ios::binary);
  myfile.write((char*) &width, sizeof(width));
  myfile.write((char*) &half_height, sizeof(half_height));
  myfile.write((char*) pixels, half_height * width * sizeof(__int16_t));
  assert(myfile.fail() == 0 && "Could not write correctly!");
  myfile.close();

  free(pixels);
}

int main(int argc, char** argv) {
  if (std::strcmp(argv[1], "julia") == 0) {
    const char* filename = argv[2];
    if (argc == 3) {
      julia_fatouCUDA(filename);
    }
    else if (argc == 5) {
      julia_fatouCUDA(filename, std::stod(argv[3]), std::stoi(argv[4]));
    }
  }
  else if (std::strcmp(argv[1], "print") == 0) {
    const char* inputFilename  = argv[2];
    const char* outputFilename = argv[3];
    if (argc == 4) {
      printimage(inputFilename, outputFilename);
    }
    else if (argc == 7) {
      printimage(
        inputFilename, outputFilename, std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]));
    }
  }
  else if (std::strcmp(argv[1], "help") == 0) {
    std::cout << "Syntax help\n===========\n";
    std::cout << "./JuliaFatou julia <outputfilename> <stepsize> <maximum iteration>\n";
    std::cout << "./JuliaFatou print <inputfilename> <outputfilename> <red> <green> <blue>\n";
    std::cout << "                   <red>, <green>, <blue> in [0,1]\n";
  }
  std::cout << "Successfully done!" << std::endl;
  return 0;
}