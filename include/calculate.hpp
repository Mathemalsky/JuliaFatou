#pragma once

#include <cstddef>

using Byte = unsigned char;

// void julia_fatouCUDA(const char* filename, const double step = 0.005, const size_t max_iter =
// 255);
void juliaFatouCUDA(Byte* textureImg);
