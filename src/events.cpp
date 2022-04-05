#include "events.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include "calculate.hpp"
#include "constants.hpp"
#include "printimage.hpp"
#include "variables.hpp"

/***********************************************************************************************************************
 *                                   events triggert by pressing keys
 **********************************************************************************************************************/

// disable gcc warning -Wunused-parameter
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
    toggleHelpWindow();
  }
  if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
    screenshot();
  }
  if (key == GLFW_KEY_F3 && action == GLFW_PRESS) {
    toggleSettingsWindow();
  }
  if (key == GLFW_KEY_R && action == GLFW_PRESS) {
    reset();
  }
  if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
    input::STATE[GLFW_KEY_UP] = true;
  }
  if (key == GLFW_KEY_UP && action == GLFW_RELEASE) {
    input::STATE[GLFW_KEY_UP] = false;
  }
  if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
    input::STATE[GLFW_KEY_DOWN] = true;
  }
  if (key == GLFW_KEY_DOWN && action == GLFW_RELEASE) {
    input::STATE[GLFW_KEY_DOWN] = false;
  }
  if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
    input::STATE[GLFW_KEY_RIGHT] = true;
  }
  if (key == GLFW_KEY_RIGHT && action == GLFW_RELEASE) {
    input::STATE[GLFW_KEY_RIGHT] = false;
  }
  if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
    input::STATE[GLFW_KEY_LEFT] = true;
  }
  if (key == GLFW_KEY_LEFT && action == GLFW_RELEASE) {
    input::STATE[GLFW_KEY_LEFT] = false;
  }
}
// enable gcc warning -Wunused-parameter
#pragma GCC diagnostic pop

void toggleHelpWindow() {
  if (imGuiWindow::SHOW_HELP_WINDOW == true) {
    imGuiWindow::SHOW_HELP_WINDOW = false;
  }
  else {
    imGuiWindow::SHOW_HELP_WINDOW = true;
  }
}

void toggleSettingsWindow() {
  if (imGuiWindow::SHOW_SETTINGS_WINDOW == true) {
    imGuiWindow::SHOW_SETTINGS_WINDOW = false;
  }
  else {
    imGuiWindow::SHOW_SETTINGS_WINDOW = true;
  }
}

using namespace functionParameters;

// set setting to standard
void reset() {
  RE_START = INITIAL_RE_START;
  IM_START = INITIAL_IM_START;
  STEP     = INITIAL_STEP;
}

// handle events that should be evaluated each frame
void handleFastEvents() {
  if (input::STATE[GLFW_KEY_UP]) {
    IM_START += control::RELATIVE_MOVE * STEP * mainWindow::HEIGHT;
  }
  if (input::STATE[GLFW_KEY_DOWN]) {
    IM_START -= control::RELATIVE_MOVE * STEP * mainWindow::HEIGHT;
  }
  if (input::STATE[GLFW_KEY_RIGHT]) {
    RE_START += control::RELATIVE_MOVE * STEP * mainWindow::WIDTH;
  }
  if (input::STATE[GLFW_KEY_LEFT]) {
    RE_START -= control::RELATIVE_MOVE * STEP * mainWindow::WIDTH;
  }
}

/***********************************************************************************************************************
 *                                   events triggert by the mouse
 **********************************************************************************************************************/
// disable gcc warning -Wunused-parameter
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  const double newStep       = STEP * std::pow(control::SCROLL_ZOOM, yoffset);
  const double stepDecrement = newStep - STEP;
  STEP                       = newStep;
  IM_START -= stepDecrement * mainWindow::HEIGHT * 0.5;
  RE_START -= stepDecrement * mainWindow::WIDTH * 0.5;
}
// enable gcc warning -Wunused-parameter
#pragma GCC diagnostic pop

/***********************************************************************************************************************
 *                                   events triggert by the window
 **********************************************************************************************************************/
// disable gcc warning -Wunused-parameter
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void windowSizeCallback(GLFWwindow* window, int width, int height) {
  mainWindow::WIDTH  = width;
  mainWindow::HEIGHT = height;
}
// enable gcc warning -Wunused-parameter
#pragma GCC diagnostic pop
