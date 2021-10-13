#include <cstring>
#include <iostream>

#include "calculate.hpp"
#include "printimage.hpp"

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
    else if (argc == 10) {
      printimage(
        inputFilename, outputFilename, std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]),
        std::stod(argv[7]), std::stod(argv[8]), std::stod(argv[9]));
    }
  }
  else if (std::strcmp(argv[1], "help") == 0) {
    std::cout << "Syntax help\n===========\n";
    std::cout << "./" << argv[0] << " julia <outputfilename> <stepsize> <maximum iteration>\n";
    std::cout
      << "./" << argv[0]
      << " print <inputfilename> <outputfilename> <red> <green> <blue> <red2> <green2> <blue2>\n";
    std::cout << "                   <red>, <green>, <blue> <red2> <green2> <blue2> are optional "
                 "arguments\n";
    std::cout << "                   with values from [0,1]\n";
    std::cout << "                   The coloring will be an interpolation between rgb and rgb2.\n";
    std::cout << "./" << argv[0] << " help\n";
  }
  else {
    std::cout << "Invalid Input!\n Type ./" << argv[0] << " for syntax help.\n";
  }
  std::cout << "Successfully done!" << std::endl;
  return 0;
}
