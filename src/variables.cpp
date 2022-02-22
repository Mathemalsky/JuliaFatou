#include "variables.hpp"

#include "constants.hpp"

namespace functionParameters {
float RE_START;
float RE_MAX;
float IM_START;
float IM_MAX;
float STEP;
float RED;
float GREEN;
float BLUE;
int MAX_ITER;
}  // namespace functionParameters

namespace settingsWindow {
bool SHOW_SETTINGS_WINDOW;
}  // namespace settingsWindow

namespace mainWindow {
unsigned int WIDTH;
unsigned int HEIGHT;

void initMainWindow() {
  WIDTH  = INITIAL_WIDTH;
  HEIGHT = INITIAL_HEIGHT;
}
}  // namespace mainWindow
