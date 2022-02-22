#pragma once

#include "types.hpp"

void* allocateGraphicsMemory();
void freeGraphicsMemory(void* cudaPixels);
void juliaFatouCUDA(Byte* textureImg, void* cudaPixels);
void singleBigFrame(Byte* pixels);
