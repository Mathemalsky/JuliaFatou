#pragma once

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

using Byte = unsigned char;
void drawJuliaFatouImage(Byte* textureImg, void* cudaPixels);
