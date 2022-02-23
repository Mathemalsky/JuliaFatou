#include "variables.hpp"

#include "constants.hpp"

namespace functionParameters {
float RE_OFFSET;
float RE_START;
float RE_MAX;
float IM_START;
float IM_MAX;
float IM_OFFSET;
float STEP;
float RED;
float GREEN;
float BLUE;
int MAX_ITER;
}  // namespace functionParameters

namespace input {
std::unordered_map<int, bool> STATE;
}  // namespace input

namespace imGuiWindow {
bool SHOW_SETTINGS_WINDOW;
bool SHOW_HELP_WINDOW;
}  // namespace imGuiWindow

namespace mainWindow {
unsigned int WIDTH;
unsigned int HEIGHT;

void initMainWindow() {
  WIDTH  = INITIAL_WIDTH;
  HEIGHT = INITIAL_HEIGHT;
}
}  // namespace mainWindow
