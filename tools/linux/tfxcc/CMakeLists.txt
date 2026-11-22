# tfxcc on the game's own shaderc build (thirdparty/pcsx2-gs/shaderc.cmake: same versions, same build options).
# nix/tfxcc.nix builds it with the sources pinned in nix/deps.nix.
cmake_minimum_required(VERSION 3.20)
project(tfxcc C CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/pcsx2-gs/shaderc.cmake")
add_executable(tfxcc tfxcc.c)
target_link_libraries(tfxcc PRIVATE shaderc)
