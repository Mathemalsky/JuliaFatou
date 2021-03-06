#include "gui.hpp"

// imgui library
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "constants.hpp"
#include "variables.hpp"

void initImGuiWindows() {
  // settings window
  imGuiWindow::SHOW_SETTINGS_WINDOW = imGuiWindow::INITIAL_SHOW_SETTINGS_WINDOW;
  functionParameters::RE_START      = functionParameters::INITIAL_RE_START;
  functionParameters::IM_START      = functionParameters::INITIAL_IM_START;
  functionParameters::RE_OFFSET     = functionParameters::INITIAL_RE_OFFSET;
  functionParameters::IM_OFFSET     = functionParameters::INITIAL_IM_OFFSET;
  functionParameters::STEP          = functionParameters::INITIAL_STEP;
  functionParameters::RED           = functionParameters::INITIAL_RED;
  functionParameters::GREEN         = functionParameters::INITIAL_GREEN;
  functionParameters::BLUE          = functionParameters::INITIAL_BLUE;
  functionParameters::MAX_ITER      = functionParameters::INITIAL_MAX_ITER;

  // help window
  imGuiWindow::SHOW_HELP_WINDOW = imGuiWindow::INITIAL_SHOW_HELP_WINDOW;
}

void setUpImgui(GLFWwindow* window, const char* glsl_version) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void) io;
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsClassic();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
}

void drawImgui() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if (imGuiWindow::SHOW_SETTINGS_WINDOW) {
    ImGui::Begin("Settings", &imGuiWindow::SHOW_SETTINGS_WINDOW);
    ImGui::SliderFloat("step size", &functionParameters::STEP, 0.0f, 1.0f);
    ImGui::SliderInt("max. iterations", &functionParameters::MAX_ITER, 0, 255);
    ImGui::SliderFloat("Re offset", &functionParameters::RE_OFFSET, -2.0f, 2.0f);
    ImGui::SliderFloat("Im offset", &functionParameters::IM_OFFSET, -2.0f, 2.0f);
    ImGui::SliderFloat("Red", &functionParameters::RED, 0.0f, 1.0f);
    ImGui::SliderFloat("Green", &functionParameters::GREEN, 0.0f, 1.0f);
    ImGui::SliderFloat("Blue", &functionParameters::BLUE, 0.0f, 1.0f);
    ImGui::Text("RE_START: %.3f", functionParameters::RE_START);
    ImGui::Text("IM_START: %.3f", functionParameters::IM_START);
    ImGui::Text(
      "Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    // if (ImGui::Button("Close")) {imGuiWindow::SHOW_SETTINGS_WINDOW = false};
    ImGui::End();
  }

  if (imGuiWindow::SHOW_HELP_WINDOW) {
    ImGui::Begin("Help", &imGuiWindow::SHOW_HELP_WINDOW);
    ImGui::Text("F1      toggle visibility of this help window");
    ImGui::Text("F2      take screenshot");
    ImGui::Text("F3      toggle visibility of settingswindow");
    ImGui::Text("LEFT    move displayed part of complex plane left,  direction -1");
    ImGui::Text("RIGHT   move displayed part of complex plane right, direction +1");
    ImGui::Text("UP      move displayed part of complex plane up,    direction +i");
    ImGui::Text("DOWN    move displayed part of complex plane down,  direction -i");
    ImGui::Text("SCROLL  zoom");
    ImGui::End();
  }

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void cleanUpImgui() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}
