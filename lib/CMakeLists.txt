set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

set(SOURCES
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  )

set(HEADERS
  ${IMGUI_DIR}/imconfig.h
  ${IMGUI_DIR}/imgui.h
  ${IMGUI_DIR}/imgui_internal.h
  ${IMGUI_DIR}/imstb_rectpack.h
  ${IMGUI_DIR}/imstb_textedit.h
  ${IMGUI_DIR}/imstb_truetype.h
  )

set(IMGUI_BACKEND_DIR ${IMGUI_DIR}/backends)

set(BACKEND_SOURCES
  ${IMGUI_BACKEND_DIR}/imgui_impl_glfw.cpp
  ${IMGUI_BACKEND_DIR}/imgui_impl_opengl3.cpp
  )

set(BACKEND_HEADERS
  ${IMGUI_BACKEND_DIR}/imgui_impl_glfw.h
  ${IMGUI_BACKEND_DIR}/imgui_impl_opengl3.h
  ${IMGUI_BACKEND_DIR}/imgui_impl_opengl3_loader.h
  )

add_library (IMGUI STATIC ${SOURCES} ${HEADERS} ${BACKEND_SOURCES} ${BACKEND_HEADERS})
target_include_directories(IMGUI PUBLIC ${IMGUI_DIR} ${IMGUI_BACKEND_DIR})

# OpenGL
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

# GLUT
find_package(GLUT REQUIRED)
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

target_link_libraries(IMGUI PUBLIC ${OPGL_LIBRARIES} -lglut -lGL)
