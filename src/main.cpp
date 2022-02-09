#include <cstring>
#include <iostream>
#include <stdio.h>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>

#include "calculate.hpp"
#include "constants.hpp"
#include "draw.hpp"
#include "gui.hpp"
#include "printimage.hpp"
#include "variables.hpp"

/*
int main(int argc, char** argv) {
  if (std::strcmp(argv[1], "julia") == 0) {
    const char* filename = argv[2];
    if (argc == 3) {
      julia_fatouCUDA(filename);
    }
    else if (argc == 5) {
      julia_fatouCUDA(filename, std::stod(argv[3]), std::stoi(argv[4]));
    }
  }
  else if (std::strcmp(argv[1], "print") == 0) {
    const char* inputFilename  = argv[2];
    const char* outputFilename = argv[3];
    if (argc == 4) {
      printimage(inputFilename, outputFilename);
    }
    else if (argc == 10) {
      printimage(
        inputFilename, outputFilename, std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]),
        std::stod(argv[7]), std::stod(argv[8]), std::stod(argv[9]));
    }
  }
  else if (std::strcmp(argv[1], "help") == 0) {
    std::cout << "Syntax help\n===========\n";
    std::cout << "./" << argv[0] << " julia <outputfilename> <stepsize> <maximum iteration>\n";
    std::cout
      << "./" << argv[0]
      << " print <inputfilename> <outputfilename> <red> <green> <blue> <red2> <green2> <blue2>\n";
    std::cout << "                   <red>, <green>, <blue> <red2> <green2> <blue2> are optional "
                 "arguments\n";
    std::cout << "                   with values from [0,1]\n";
    std::cout << "                   The coloring will be an interpolation between rgb and rgb2.\n";
    std::cout << "./" << argv[0] << " help\n";
  }
  else {
    std::cout << "Invalid Input!\n Type ./" << argv[0] << " for syntax help.\n";
  }
  std::cout << "Successfully done!" << std::endl;
  return 0;
}
*/

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int, char**) {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  const char* glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

  // create window in specified size
  GLFWwindow* window = glfwCreateWindow(
    mainWindow::INITIAL_WIDTH, mainWindow::INITIAL_HEIGHT, mainWindow::NAME, NULL, NULL);
  if (window == NULL)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // enable vsync

  // enable the texture which will be drawn
  glEnable(GL_TEXTURE_2D);
  glLoadIdentity();

  // set initial state of the settings window
  initSettingsWindow();

  // setup Dear ImGui
  setUpImgui(window, glsl_version);

  // allocate memory for the drawing
  const unsigned int textureSize =
    universal::RGB_COLORS * mainWindow::INITIAL_WIDTH * mainWindow::INITIAL_HEIGHT;
  Byte* textureImg = (Byte*) malloc(textureSize);

  // main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // can be replaced by global window variables
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);

    // draw the julia fatou image
    drawJuliaFatouImage(textureImg);

    // draw the imgui over the fatou image
    drawImgui();

    // swap the drawings to the displayed frame
    glfwSwapBuffers(window);
  }

  // free the memory for the texture
  free(textureImg);

  // clean up Dear ImGui
  cleanUpImgui();

  // clean up glfw window
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
