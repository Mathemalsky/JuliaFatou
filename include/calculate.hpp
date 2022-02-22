#pragma once

using Byte = unsigned char;

void* allocateGraphicsMemory();
void freeGraphicsMemory(void* cudaPixels);
void juliaFatouCUDA(Byte* textureImg, void* cudaPixels);
void screenshot(Byte* pixels);
