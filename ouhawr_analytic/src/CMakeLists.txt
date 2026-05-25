cmake_minimum_required(VERSION 3.16)
project(OUHAWR_MSDHMM VERSION 1.0 LANGUAGES CXX)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wpedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG -march=native)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
endif()

# ── Source files ──────────────────────────────────────────────────────────────
set(LIB_SOURCES
    src/Preprocessing.cpp
    src/StreamHMM.cpp
    src/Distributions.cpp
    src/Decoder.cpp
    src/EmbeddedTrainer.cpp
    src/Database.cpp
)

# ── Library ───────────────────────────────────────────────────────────────────
add_library(ouhawr STATIC ${LIB_SOURCES})
target_include_directories(ouhawr PUBLIC src/)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(ouhawr_test src/main.cpp)
target_link_libraries(ouhawr_test PRIVATE ouhawr)

# ── Link math library on Linux ────────────────────────────────────────────────
if(UNIX)
    target_link_libraries(ouhawr PRIVATE m)
endif()

# ── Optional: OpenMP for parallelism ─────────────────────────────────────────
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ouhawr PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS ouhawr_test DESTINATION bin)
install(TARGETS ouhawr       DESTINATION lib)
install(FILES
    src/MSDHMM.hpp
    src/Preprocessing.hpp
    src/Database.hpp
    DESTINATION include/ouhawr)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "━━━ OUHAWR MSDHMM Build Summary ━━━")
message(STATUS "  Sources  : ${LIB_SOURCES}")
message(STATUS "  Target   : ouhawr_test")
message(STATUS "  C++      : ${CMAKE_CXX_STANDARD}")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
