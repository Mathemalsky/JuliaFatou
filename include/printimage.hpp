#pragma once

using Byte = unsigned char;

void printimage(
  const char* inputFilename, const char* outputFilename, const double red = 1, const double green = 0,
  const double blue = 1, const double red2 = 1, const double green2 = 1, const double blue2 = 0);

void printImage(const char* filename, Byte* pixels, const unsigned int width, const unsigned int height);
