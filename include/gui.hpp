#pragma once

#include <GLFW/glfw3.h>

void initSettingsWindow();
void setUpImgui(GLFWwindow* window, const char* glsl_version);
void drawImgui();
void cleanUpImgui();
