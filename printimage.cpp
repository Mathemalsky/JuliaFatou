#include "printimage.hpp"

#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>

const size_t BYTES_PER_PIXEL  = 3;  /// red, green, & blue
const size_t FILE_HEADER_SIZE = 14;
const size_t INFO_HEADER_SIZE = 40;

void generateBitmapImage(
  unsigned char* image, size_t height, size_t width, const char* imageFileName);
unsigned char* createBitmapFileHeader(int height, int stride);
unsigned char* createBitmapInfoHeader(int height, int width);

void generateBitmapImage(
  unsigned char* image, size_t height, size_t width, const char* imageFileName) {
  size_t widthInBytes = width * BYTES_PER_PIXEL;

  unsigned char padding[3] = {0, 0, 0};
  int paddingSize          = (4 - (widthInBytes) % 4) % 4;

  size_t stride = (widthInBytes) + paddingSize;

  FILE* imageFile = fopen(imageFileName, "wb");

  unsigned char* fileHeader = createBitmapFileHeader(height, stride);
  fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

  unsigned char* infoHeader = createBitmapInfoHeader(height, width);
  fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

  for (size_t i = 0; i < height; i++) {
    fwrite(image + (i * widthInBytes), BYTES_PER_PIXEL, width, imageFile);
    fwrite(padding, 1, paddingSize, imageFile);
  }

  fclose(imageFile);
}

unsigned char* createBitmapFileHeader(int height, int stride) {
  int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

  static unsigned char fileHeader[] = {
    0, 0,        /// signature
    0, 0, 0, 0,  /// image file size in bytes
    0, 0, 0, 0,  /// reserved
    0, 0, 0, 0,  /// start of pixel array
  };

  fileHeader[0]  = (unsigned char) ('B');
  fileHeader[1]  = (unsigned char) ('M');
  fileHeader[2]  = (unsigned char) (fileSize);
  fileHeader[3]  = (unsigned char) (fileSize >> 8);
  fileHeader[4]  = (unsigned char) (fileSize >> 16);
  fileHeader[5]  = (unsigned char) (fileSize >> 24);
  fileHeader[10] = (unsigned char) (FILE_HEADER_SIZE + INFO_HEADER_SIZE);

  return fileHeader;
}

unsigned char* createBitmapInfoHeader(int height, int width) {
  static unsigned char infoHeader[] = {
    0, 0, 0, 0,  /// header size
    0, 0, 0, 0,  /// image width
    0, 0, 0, 0,  /// image height
    0, 0,        /// number of color planes
    0, 0,        /// bits per pixel
    0, 0, 0, 0,  /// compression
    0, 0, 0, 0,  /// image size
    0, 0, 0, 0,  /// horizontal resolution
    0, 0, 0, 0,  /// vertical resolution
    0, 0, 0, 0,  /// colors in color table
    0, 0, 0, 0,  /// important color count
  };

  infoHeader[0]  = (unsigned char) (INFO_HEADER_SIZE);
  infoHeader[4]  = (unsigned char) (width);
  infoHeader[5]  = (unsigned char) (width >> 8);
  infoHeader[6]  = (unsigned char) (width >> 16);
  infoHeader[7]  = (unsigned char) (width >> 24);
  infoHeader[8]  = (unsigned char) (height);
  infoHeader[9]  = (unsigned char) (height >> 8);
  infoHeader[10] = (unsigned char) (height >> 16);
  infoHeader[11] = (unsigned char) (height >> 24);
  infoHeader[12] = (unsigned char) (1);
  infoHeader[14] = (unsigned char) (BYTES_PER_PIXEL * 8);

  return infoHeader;
}

// read the input data
int16_t* readimage(size_t& heigth, size_t& width, const char* filename) {
  //myfile >> heigth;
  //myfile >> width;
  size_t half_height = 0;  // unbedingt ändern
  int16_t* pixels = (int16_t*)malloc(half_height * width);
  FILE* myfile = fopen(filename, "rb");
  size_t size = fread(pixels, sizeof(int16_t), half_height * width, myfile);
  fclose(myfile);
  // size noch mit assert abfangen
  return pixels;
}

void printimage(const char *filename) {
  // image_name, max_iter, color may become arguments in the future
  int maxiter         = 50;
  const char* image_name = "testimg.bmp";

  const double red   = 0.8;
  const double green = 0;
  const double blue  = 0.8;

  size_t half_height, width;
  const int16_t* pixels = readimage(half_height, width, filename);

  const size_t half_size = half_height * width;
  const size_t size      = 2 * half_size;
  maxiter                = 0;
  for(size_t i=0; i<half_size; ++i) {
    if (pixels[i] > maxiter) {
      maxiter = pixels[i];
    }
  }

  unsigned char image[size][BYTES_PER_PIXEL];

  for (size_t i = 0; i < half_size; ++i) {
    double intensity          = double(pixels[i]) / maxiter * 255;
    unsigned char pixel_red   = std::round(red * intensity);    // red
    unsigned char pixel_blue  = std::round(blue * intensity);   // blue
    unsigned char pixel_green = std::round(green * intensity);  // green
    image[i][2]               = pixel_red;
    image[i][1]               = pixel_green;
    image[i][0]               = pixel_blue;
    // abuse invariance under 180 degree rotaion
    image[size - 1 - i][2] = pixel_red;
    image[size - 1 - i][1] = pixel_green;
    image[size - 1 - i][0] = pixel_blue;
  }

  generateBitmapImage((unsigned char*) image, 2 * half_height, width, image_name);
  std::cout << "maximum Iterations: " << maxiter << std::endl;
}
