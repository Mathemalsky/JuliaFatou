#pragma once

#include <unordered_map>

namespace functionParameters {
extern float IM_OFFSET;
extern float RE_OFFSET;
extern double RE_START;
extern double IM_START;
extern double STEP;
extern float D_RED;
extern float D_GREEN;
extern float D_BLUE;
extern float C_RED;
extern float C_GREEN;
extern float C_BLUE;
extern int MAX_ITER;
}  // namespace functionParameters

namespace input {
extern std::unordered_map<int, bool> STATE;
extern double MOUSE_X;
extern double MOUSE_Y;
extern bool MOUSE_USE;
}  // namespace input

namespace imGuiWindow {
extern bool SHOW_SETTINGS_WINDOW;
extern bool SHOW_HELP_WINDOW;
extern bool CALC_CONVERGENCE;
}  // namespace imGuiWindow

namespace mainWindow {
extern unsigned int WIDTH;
extern unsigned int HEIGHT;
void initMainWindow();
}  // namespace mainWindow
