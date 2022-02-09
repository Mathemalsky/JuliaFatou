#include "gui.hpp"

// imgui library
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "constants.hpp"
#include "variables.hpp"

void initSettingsWindow() {
  settingsWindow::SHOW_SETTINGS_WINDOW = settingsWindow::INITIAL_SHOW_SETTINGS_WINDOW;
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

  if (settingsWindow::SHOW_SETTINGS_WINDOW) {
    ImGui::Begin("Settings", &settingsWindow::SHOW_SETTINGS_WINDOW);
    ImGui::SliderFloat("step size", &functionParameters::STEP, 0.0f, 1.0f);
    ImGui::SliderInt("max. iterations", &functionParameters::MAX_ITER, 0, 255);
    ImGui::Text(
      "Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
      ImGui::GetIO().Framerate);
    if (ImGui::Button("Close"))
      settingsWindow::SHOW_SETTINGS_WINDOW = false;
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
