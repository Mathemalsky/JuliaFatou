#pragma once

#include "types.hpp"

namespace control {
inline const double RELATIVE_MOVE = 0.01;
inline const double SCROLL_ZOOM   = 0.98;
}  // namespace control

namespace mainWindow {
inline const char* NAME                      = "JuliaFatou";
inline const unsigned int INITIAL_WIDTH      = 1560;
inline const unsigned int INITIAL_HEIGHT     = 960;
inline const unsigned int MAX_WIDTH_X_HEIGHT = 1920 * 1080;
}  // namespace mainWindow

namespace settingsGPU {
inline const unsigned int BLOCKSIZE = 256;
}  // namespace settingsGPU

namespace imGuiWindow {
inline const bool INITIAL_SHOW_SETTINGS_WINDOW = true;
inline const bool INITIAL_SHOW_HELP_WINDOW     = true;
inline const bool INITIAL_CALC_CONVERGENCE     = true;
inline const bool INITIAL_MOUSE_USE            = true;
}  // namespace imGuiWindow

namespace universal {
inline const unsigned int RGB_COLORS = 3;
inline const Byte MAX_BYTE           = 255;
}  // namespace universal

namespace functionParameters {
inline const float INITIAL_RE_OFFSET        = -0.78f;
inline const float INITIAL_IM_OFFSET        = -0.18f;
inline const float INITIAL_D_RED            = 0.8f;
inline const float INITIAL_D_GREEN          = 0.0f;
inline const float INITIAL_D_BLUE           = 0.6f;
inline const float INITIAL_C_RED            = 0.2f;
inline const float INITIAL_C_GREEN          = 1.0f;
inline const float INITIAL_C_BLUE           = 0.4f;
inline const double INITIAL_RE_START        = -2.0;
inline const double INITIAL_IM_START        = -2.0 * mainWindow::INITIAL_HEIGHT / mainWindow::INITIAL_WIDTH;
inline const double INITIAL_STEP            = -2.0f * INITIAL_RE_START / mainWindow::INITIAL_WIDTH;
inline const double CONVERGENCE_LIMIT       = 0.000001f;
inline const double NORM_LIMIT              = 1000;
inline const unsigned int SCREENSHOT_WIDTH  = 3840;
inline const unsigned int SCREENSHOT_HEIGHT = 2160;
inline const int INITIAL_MAX_ITER           = 70;
}  // namespace functionParameters
