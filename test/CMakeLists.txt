cmake_minimum_required(VERSION 3.10)

# This must be done before any language is set
# (ie: before any project() or enable_language() command).
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

project(demo)

set(CMAKE_CXX_STANDARD 14)

# this statement should be written before add_subdirectory
# global include, transmitted downward along the directories 
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(cuda)

add_executable(${PROJECT_NAME} main.cpp)

set(EXTRA_LIBS ${EXTRA_LIBS} ${mylib})
target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS})

enable_testing()

add_test(
    NAME TestGPU
    COMMAND ${PROJECT_NAME}
)
set_tests_properties(TestGPU PROPERTIES DEPENDS ${PROJECT_NAME}
                        ENVIRONMENT "CUDA_VISIBLE_DEVICES=4,5,6,7")

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)

add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD 
                   COMMAND ${CMAKE_CTEST_COMMAND} --verbose
                   VERBATIM)
