#pragma once

namespace functionParameters {
inline const float INITIAL_RE_START = -2.0f;
inline const float INITIAL_IM_START = -2.0f * 720 / 1280;
inline const float INITIAL_STEP     = 4.0f / 1280;
inline const float INITIAL_RED      = 0.8f;
inline const float INITIAL_GREEN    = 0.0f;
inline const float INITIAL_BLUE     = 0.6f;
inline const int INITIAL_MAX_ITER   = 70;
inline const double NORM_LIMIT      = 1000;
}  // namespace functionParameters

namespace mainWindow {
inline const char* NAME                  = "JuliaFatou";
inline const unsigned int INITIAL_HEIGHT = 720;
inline const unsigned int INITIAL_WIDTH  = 1280;
}  // namespace mainWindow

namespace settingsGPU {
inline const unsigned int BLOCKSIZE = 256;
}  // namespace settingsGPU

namespace settingsWindow {
inline const bool INITIAL_SHOW_SETTINGS_WINDOW = true;
}  // namespace settingsWindow

namespace universal {
inline const unsigned int RGB_COLORS = 3;
}  // namespace universal
