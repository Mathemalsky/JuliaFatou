#pragma once

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

void toggleHelpWindow();
void toggleSettingsWindow();
void reset();
void handleFastEvents(GLFWwindow* window);

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

void windowSizeCallback(GLFWwindow* window, int width, int height);
