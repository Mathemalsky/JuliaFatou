#include <cstring>
#include <iostream>
#include <cstdio>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "calculate.hpp"
#include "constants.hpp"
#include "draw.hpp"
#include "events.hpp"
#include "gui.hpp"
#include "printimage.hpp"
#include "variables.hpp"

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
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);  // 3.2+ only
                                                                    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                                                                    // GL_TRUE);            // 3.0+ only
#endif

  // create window in specified size
  GLFWwindow* window =
    glfwCreateWindow(mainWindow::INITIAL_WIDTH, mainWindow::INITIAL_HEIGHT, mainWindow::NAME, NULL, NULL);
  if (window == NULL)
    return 1;
  glfwMakeContextCurrent(window);
  gladLoadGL();
  glfwSwapInterval(1);  // enable vsync

  // set initial state of the settings window
  initSettingsWindow();
  mainWindow::initMainWindow();

  // setup Dear ImGui
  setUpImgui(window, glsl_version);

  // allocate memory for the drawing
  const unsigned int textureSize = universal::RGB_COLORS * mainWindow::INITIAL_WIDTH * mainWindow::INITIAL_HEIGHT;
  Byte* textureImg               = (Byte*) malloc(textureSize);
  void* cudaPixels               = allocateGraphicsMemory();

  // enable the texture which will be drawn
  glEnable(GL_TEXTURE_2D);
  glActiveTexture(GL_TEXTURE0);
  glMatrixMode(GL_PROJECTION);
  glOrtho(0, 0, -1.f, 1.f, 1.f, -1.f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // get an ID for the texture
  GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D(
    GL_TEXTURE_2D, 0, GL_RGB, mainWindow::INITIAL_WIDTH, mainWindow::INITIAL_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE,
    textureImg);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glViewport(0, 0, mainWindow::WIDTH, mainWindow::HEIGHT);

  // Test
  glfwSetKeyCallback(window, keyCallback);

  // main loop
  while (!glfwWindowShouldClose(window)) {
    // runs only through the loop if something changed
    glfwPollEvents();

    // handle Events triggert by user input, like keyboard etc.
    handleFastEvents();

    // draw the julia fatou image
    drawJuliaFatouImage(textureImg, cudaPixels);

    // draw the imgui over the fatou image
    drawImgui();

    // swap the drawings to the displayed frame
    glfwSwapBuffers(window);
  }

  // free the memory for the texture
  free(textureImg);
  freeGraphicsMemory(cudaPixels);

  // clean up Dear ImGui
  cleanUpImgui();

  // clean up glfw window
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
