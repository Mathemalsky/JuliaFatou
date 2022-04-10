#include "variables.hpp"

#include "constants.hpp"

namespace functionParameters {
float RE_OFFSET;
float RE_START;
float IM_START;
float IM_OFFSET;
float STEP;
float C_RED;
float C_GREEN;
float C_BLUE;
float D_RED;
float D_GREEN;
float D_BLUE;
int MAX_ITER;
}  // namespace functionParameters

namespace input {
std::unordered_map<int, bool> STATE;
}  // namespace input

namespace imGuiWindow {
bool SHOW_SETTINGS_WINDOW;
bool SHOW_HELP_WINDOW;
bool CALC_CONVERGENCE;
}  // namespace imGuiWindow

namespace mainWindow {
unsigned int WIDTH;
unsigned int HEIGHT;

void initMainWindow() {
  WIDTH  = INITIAL_WIDTH;
  HEIGHT = INITIAL_HEIGHT;
}
}  // namespace mainWindow
