cmake_minimum_required(VERSION 3.20)

project(running_ai_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(third_party/pybind11)

add_library(running_training_core STATIC
    src/course_rules.cpp
    src/physics_backend_loader.cpp
    src/reward_profiles.cpp
    src/training_env.cpp
)
set_target_properties(running_training_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(running_training_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
if(MSVC)
    target_compile_definitions(running_training_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

pybind11_add_module(_running_env src/python_module.cpp)
target_link_libraries(_running_env PRIVATE running_training_core)

install(TARGETS _running_env
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
)
