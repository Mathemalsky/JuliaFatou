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
    else if (argc == 7) {
      printimage(
        inputFilename, outputFilename, std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]));
    }
  }
  else if (std::strcmp(argv[1], "help") == 0) {
    std::cout << "Syntax help\n===========\n";
    std::cout << "./JuliaFatou julia <outputfilename> <stepsize> <maximum iteration>\n";
    std::cout << "./JuliaFatou print <inputfilename> <outputfilename> <red> <green> <blue>\n";
    std::cout << "                   <red>, <green>, <blue> are optional arguments with values "
                 "from [0,1]\n";
  }
  std::cout << "Successfully done!" << std::endl;
  return 0;
}
