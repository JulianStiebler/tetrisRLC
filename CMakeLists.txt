cmake_minimum_required(VERSION 3.10)

project(tetris VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

include_directories(include)

file(GLOB SOURCES "src/*.c")

add_library(api SHARED ${SOURCES})

if(WIN32)
    target_compile_definitions(api PRIVATE TETRIS_EXPORTS)
    set_target_properties(api PROPERTIES SUFFIX ".dll")
else()
    set_target_properties(api PROPERTIES PREFIX "lib")
    set_target_properties(api PROPERTIES SUFFIX ".so")
endif()

add_executable(tetris_exe ${SOURCES})