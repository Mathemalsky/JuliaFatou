#include "draw.hpp"

#include <GLFW/glfw3.h>

void drawJuliaFatouImage() {
  glColor3f(1.0f, 0.0f, 0.0f);
  glBegin(GL_TRIANGLES);  // begin triangle coordinates
  glVertex2f(-0.5f, 0.5f);
  glVertex2f(-1.0f, 0.15f);
  glVertex2f(-0.15f, 0.5f);
  glEnd();  // end triangle coordinates
}
