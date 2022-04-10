#pragma once

#include "types.hpp"

namespace control {
inline const double RELATIVE_MOVE = 0.01;
inline const double SCROLL_ZOOM   = 0.98;
}  // namespace control

namespace functionParameters {
inline const float CONVERGENCE_LIMIT        = 0.001f;
inline const float INITIAL_RE_START         = -2.0f;
inline const float INITIAL_IM_START         = -2.0f * 720 / 1280;
inline const float INITIAL_RE_OFFSET        = -0.78f;
inline const float INITIAL_IM_OFFSET        = -0.18f;
inline const float INITIAL_STEP             = 4.0f / 1280;
inline const float INITIAL_D_RED            = 0.8f;
inline const float INITIAL_D_GREEN          = 0.0f;
inline const float INITIAL_D_BLUE           = 0.6f;
inline const float INITIAL_C_RED            = 0.2f;
inline const float INITIAL_C_GREEN          = 1.0f;
inline const float INITIAL_C_BLUE           = 0.4f;
inline const int INITIAL_MAX_ITER           = 70;
inline const double NORM_LIMIT              = 1000;
inline const unsigned int SCREENSHOT_WIDTH  = 3840;
inline const unsigned int SCREENSHOT_HEIGHT = 2160;
}  // namespace functionParameters

namespace mainWindow {
inline const char* NAME                      = "JuliaFatou";
inline const unsigned int INITIAL_WIDTH      = 1280;
inline const unsigned int INITIAL_HEIGHT     = 720;
inline const unsigned int MAX_WIDTH_X_HEIGHT = 1920 * 1080;
}  // namespace mainWindow

namespace settingsGPU {
inline const unsigned int BLOCKSIZE = 256;
}  // namespace settingsGPU

namespace imGuiWindow {
inline const bool INITIAL_SHOW_SETTINGS_WINDOW = true;
inline const bool INITIAL_SHOW_HELP_WINDOW     = true;
inline const bool INITIAL_CALC_CONVERGENCE = true;
}  // namespace imGuiWindow

namespace universal {
inline const unsigned int RGB_COLORS = 3;
inline const Byte MAX_BYTE           = 255;
}  // namespace universal
