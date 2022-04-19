#include "variables.hpp"

#include "constants.hpp"

namespace functionParameters {
float RE_OFFSET;
float IM_OFFSET;
double RE_START;
double IM_START;
double STEP;
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
double MOUSE_X;
double MOUSE_Y;
bool MOUSE_USE;
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
