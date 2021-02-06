#include <assert.h>
#include <complex>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stddef.h>
#include <vector>

std::complex<double> function(std::complex<double>& z) {
    return z*z - std::complex<double>(-0.6, 0.6);
}

int main() {
    const double step = 0.002;
    const std::complex<double> start(-2,-2);
    const size_t max_iter = 70;
    const double norm_limit = 4;

    const size_t width = std::abs(double(start.real()*2/step));
    const size_t half_height = std::abs(double(start.imag()/step));           // calculate just one half because of rotation symmetrie
    std::vector<int> pixels(width * half_height);

    for(size_t i=0; i<half_height; ++i) {
        for(size_t j=0; j<width; ++j) {
            std::complex<double> z(start.real() + step*j, start.real() +step*i);
            size_t counter = 0;
            do {
                z = function(z);
                ++counter;
            } while(std::abs(z) < norm_limit && counter <= max_iter);
            pixels.at(j + i * width) = counter;
        }
    }

    std::ofstream myfile("image.txt", std::ios::binary);
    myfile << half_height << "\n" << width << "\n";
    for(auto &pixel : pixels) {
        myfile << pixel <<"\n";
    }

    std::cout << "Successfully done!" << std::endl;
    return 0;
}
