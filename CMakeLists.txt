cmake_minimum_required(VERSION 3.12.0)
project(ELSED)

set(CMAKE_CXX_STANDARD 17)
if (MSVC)
    add_compile_options(/MP /utf-8 /bigobj)
endif()

find_package(OpenCV REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_library(ELSED STATIC
        src/FullSegmentInfo.cpp
        src/ELSED.cpp
        src/EdgeDrawer.cpp)
target_include_directories(ELSED
        PRIVATE src
        PUBLIC include
        PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ELSED PUBLIC ${OpenCV_LIBS})

#################################################
#                Build Examples                 #
#################################################
set(BUILD_EXAMPLE FALSE)
if(BUILD_EXAMPLE)
add_executable(ELSED_main example/main.cpp)
target_include_directories(ELSED_main PRIVATE src include)
target_link_libraries(ELSED_main PRIVATE ELSED)

add_executable(ELSED_Line_main example/LineDetectMain.cpp)
target_include_directories(ELSED_Line_main PRIVATE src include)
target_link_libraries(ELSED_Line_main PRIVATE ELSED)
endif()