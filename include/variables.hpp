#pragma once

#include <unordered_map>

namespace functionParameters {
extern float RE_START;
extern float RE_MAX;
extern float IM_START;
extern float IM_MAX;
extern float STEP;
extern float RED;
extern float GREEN;
extern float BLUE;
extern int MAX_ITER;
}  // namespace functionParameters

namespace input {
extern std::unordered_map<int, bool> STATE;
}  // namespace input

namespace mainWindow {
extern unsigned int WIDTH;
extern unsigned int HEIGHT;
void initMainWindow();
}  // namespace mainWindow

namespace imGuiWindow {
extern bool SHOW_SETTINGS_WINDOW;
extern bool SHOW_HELP_WINDOW;
}  // namespace imGuiWindow
