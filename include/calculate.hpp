#pragma once

#include <cstddef>

using Byte = unsigned char;

void juliaFatouCUDA(Byte* textureImg, void* cudaPixels);
void* allocateGraphicsMemory();
void freeGraphicsMemory(void* cudaPixels);
