cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(marcel CXX)
set(CMAKE_CXX_STANDARD 11)

# Use OpenMP

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# Use cubature

add_subdirectory(cubature)

# Add current directory to search path

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add the executables

set(SOURCE_FILES "test_integrate.cpp")
add_executable(test_integrate ${SOURCE_FILES})
set_source_files_properties(${SOURCE_FILES}
  PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Weffc++ -Wpedantic")

target_link_libraries(test_integrate cubature)
