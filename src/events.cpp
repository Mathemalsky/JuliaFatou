#include "events.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include "calculate.hpp"
#include "constants.hpp"
#include "printimage.hpp"
#include "variables.hpp"

// disable gcc warning -Wunused-parameter
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
    screenshot();
  }
}
// enable gcc warning -Wunused-parameter
#pragma GCC diagnostic pop

void handleFastEvents() {
}
