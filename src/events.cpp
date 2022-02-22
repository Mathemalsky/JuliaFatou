#include "events.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include "calculate.hpp"
#include "constants.hpp"
#include "printimage.hpp"
#include "variables.hpp"

using namespace functionParameters;

static inline Byte doubleToByte(const double a) {
  return std::round(a * universal::MAX_BYTE);
}

static inline std::string byteToHexstring(const Byte byte) {
  const std::string table = "0123456789abcdef";
  std::string hex;
  hex.reserve(2);
  hex.push_back(table[byte / 16]);
  hex.push_back(table[byte % 16]);
  return hex;
}

static inline std::string toHexstring(const double a) {
  return byteToHexstring(doubleToByte(a));
}

// disable gcc warning -Wunused-parameter
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
    std::string filename = std::to_string(functionParameters::SCREENSHOT_WIDTH) + "x";
    filename += std::to_string(functionParameters::SCREENSHOT_HEIGHT) + "_";
    filename += std::to_string(RE_START) + "_" + std::to_string(IM_START) + "_" + std::to_string(STEP) + "_";
    filename += toHexstring(RED) + toHexstring(GREEN) + toHexstring(BLUE) + ".png";

    // Error here calloc has to be changed to malloc.
    // call to screenshot seems to have no effect on the data
    Byte* pixels = (Byte*) std::calloc(
      universal::RGB_COLORS * functionParameters::SCREENSHOT_WIDTH * functionParameters::SCREENSHOT_HEIGHT, 1);
    screenshot(pixels);

    printImage(filename.c_str(), pixels, functionParameters::SCREENSHOT_WIDTH, functionParameters::SCREENSHOT_HEIGHT);
    std::cout << "Current image has been saved as <" + filename + ">\n";
    free(pixels);
  }
}
// enable gcc warning -Wunused-parameter
#pragma GCC diagnostic pop

// see https://stackoverflow.com/questions/46631814/handling-multiple-keys-input-at-once-with-glfw
class Input {
private:
public:
};
