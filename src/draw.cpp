#include "draw.hpp"

#include <cmath>
#include <cstdlib>

#include "calculate.hpp"
#include "constants.hpp"
#include "variables.hpp"

// See also https://learnopengl.com/Getting-started/Shaders and
// https://stackoverflow.com/questions/21070076/opengl-generating-a-2d-texture-from-a-data-array-to-display-on-a-quad
// adjust this function to the desired functionality, move malloc and free out of the loop
void drawJuliaFatouImage(Byte* textureImg, void* cudaPixels) {
  // bringing the data from the GPU to the texture
  juliaFatouCUDA(textureImg, cudaPixels);

  // bringing the data from the texture back to GPU
  glTexSubImage2D(
    GL_TEXTURE_2D, 0, 0, 0, mainWindow::INITIAL_WIDTH, mainWindow::INITIAL_HEIGHT, GL_RGB,
    GL_UNSIGNED_BYTE, textureImg);

  // map the texture to the entire window
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0);
  glVertex2d(-1.0, -1.0);
  glTexCoord2d(1.0, 0.0);
  glVertex2d(1.0, -1.0);
  glTexCoord2d(1.0, 1.0);
  glVertex2d(1.0, 1.0);
  glTexCoord2d(0.0, 1.0);
  glVertex2d(-1.0, 1.0);
  glEnd();
}
