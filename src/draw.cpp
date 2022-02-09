#include "draw.hpp"

#include <cmath>
#include <cstdlib>

#include <GLFW/glfw3.h>

#include "constants.hpp"
#include "variables.hpp"

using Byte = unsigned char;

// See also https://learnopengl.com/Getting-started/Shaders and
// https://stackoverflow.com/questions/21070076/opengl-generating-a-2d-texture-from-a-data-array-to-display-on-a-quad
// adjust this function to the desired functionality, move malloc and free out of the loop
void drawJuliaFatouImage() {
  const unsigned int textureSize =
    universal::RGB_COLORS * mainWindow::INITIAL_WIDTH * mainWindow::INITIAL_HEIGHT;

  // allocate memory for the texture
  // IMPROVEMENT: DO NOT ALLOCATE INSIDE THE LOOP
  Byte* textureImg = (Byte*) malloc(textureSize);

  // REPLACE BY PROPPER FUNCTION FOR DRAWING
  for (unsigned int i = 0; i < textureSize; i++) {
    textureImg[i] = std::round(functionParameters::STEP * 255);
  }

  // gte an ID for the texture
  GLuint textureID;

  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);

  glTexImage2D(
    GL_TEXTURE_2D, 0, GL_RGB, mainWindow::INITIAL_WIDTH, mainWindow::INITIAL_HEIGHT, 0, GL_RGB,
    GL_UNSIGNED_BYTE, textureImg);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  glBindTexture(GL_TEXTURE_2D, textureID);

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

  // IMPROVEMENT: MOVE FREE OUT OF THE LOOP
  free(textureImg);
}
