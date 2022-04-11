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
  functionParameters::D_RED         = functionParameters::INITIAL_D_RED;
  functionParameters::D_GREEN       = functionParameters::INITIAL_D_GREEN;
  functionParameters::D_BLUE        = functionParameters::INITIAL_D_BLUE;
  functionParameters::C_RED         = functionParameters::INITIAL_C_RED;
  functionParameters::C_GREEN       = functionParameters::INITIAL_C_GREEN;
  functionParameters::C_BLUE        = functionParameters::INITIAL_C_BLUE;
  functionParameters::MAX_ITER      = functionParameters::INITIAL_MAX_ITER;

  // help window
  imGuiWindow::SHOW_HELP_WINDOW = imGuiWindow::INITIAL_SHOW_HELP_WINDOW;
  imGuiWindow::CALC_CONVERGENCE = imGuiWindow::INITIAL_CALC_CONVERGENCE;
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
    ImGui::Text("Set general settings:");
    ImGui::SliderInt("max. iterations", &functionParameters::MAX_ITER, 0, 255);
    ImGui::SliderFloat("Re offset", &functionParameters::RE_OFFSET, -2.0f, 2.0f);
    ImGui::SliderFloat("Im offset", &functionParameters::IM_OFFSET, -2.0f, 2.0f);
    ImGui::Text("Select color for divergent points:");
    ImGui::SliderFloat("divergent Red", &functionParameters::D_RED, 0.0f, 1.0f);
    ImGui::SliderFloat("divergent Green", &functionParameters::D_GREEN, 0.0f, 1.0f);
    ImGui::SliderFloat("divergent Blue", &functionParameters::D_BLUE, 0.0f, 1.0f);
    ImGui::Checkbox("Consider convergence", &imGuiWindow::CALC_CONVERGENCE);
    ImGui::Text("Select color for convergent points:");
    ImGui::SliderFloat("convergent Red", &functionParameters::C_RED, 0.0f, 1.0f);
    ImGui::SliderFloat("convergent Green", &functionParameters::C_GREEN, 0.0f, 1.0f);
    ImGui::SliderFloat("convergent Blue", &functionParameters::C_BLUE, 0.0f, 1.0f);
    ImGui::Text("Display info:\nstep size %.6f\n", functionParameters::STEP);
    ImGui::Text(
      "Area: %.6f < x < %.6f\n      %.6f < y < %.6f", functionParameters::RE_START,
      functionParameters::RE_START + functionParameters::STEP * mainWindow::WIDTH, functionParameters::IM_START,
      functionParameters::IM_START + functionParameters::STEP * mainWindow::HEIGHT);
    ImGui::Text(
      "Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    // if (ImGui::Button("Close")) {imGuiWindow::SHOW_SETTINGS_WINDOW = false};
    ImGui::Text("Mouse x: %.f\nMouse y: %.f", input::MOUSE_X, input::MOUSE_Y);
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
