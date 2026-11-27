# GHR's unit tests (tests/unit/unit_tests.cpp): one plain program, registered with CTest. It links only the pure-logic
# sources it tests, so it needs neither the recompiled game code nor a display.
#
# From the top level: -DKAMITE_BUILD_TESTS=ON, then ctest. On its own (no game code at all):
#   cmake -S tests/unit -B build/unit -G Ninja && ninja -C build/unit && ctest --test-dir build/unit
cmake_minimum_required(VERSION 3.21)
if(NOT PROJECT_NAME)
    project(GhrUnitTests LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    enable_testing()
endif()

set(KAMITE_UNIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
cmake_path(NORMAL_PATH KAMITE_UNIT_ROOT)

# The tar.gz tests need raylib's inflate (sinfl.h), the one GHR's updater uses. In the full build it is already fetched
# (runtime/ps2xRuntime); on its own, only its source is fetched (same release, nothing of it is built).
include(FetchContent)
FetchContent_GetProperties(raylib)
if(NOT raylib_POPULATED AND NOT raylib_SOURCE_DIR)
    FetchContent_Declare(raylib_unit_tests
        GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
        GIT_TAG "5.5"
        SOURCE_SUBDIR "does-not-exist") # source only: no add_subdirectory
    FetchContent_MakeAvailable(raylib_unit_tests)
    set(KAMITE_UNIT_SINFL_DIR "${raylib_unit_tests_SOURCE_DIR}/src/external")
else()
    set(KAMITE_UNIT_SINFL_DIR "${raylib_SOURCE_DIR}/src/external")
endif()

add_executable(ghr_unit_tests
    unit_tests.cpp
    sinfl_impl.c
    "${KAMITE_UNIT_ROOT}/src/kamite/paths.cpp"
    "${KAMITE_UNIT_ROOT}/src/kamite/portal_file_chooser.cpp"
    "${KAMITE_UNIT_ROOT}/src/kamite/steam_deck.cpp"
    "${KAMITE_UNIT_ROOT}/src/kamite/steam_shortcuts.cpp"
    "${KAMITE_UNIT_ROOT}/src/kamite/update_archive.cpp")
target_include_directories(ghr_unit_tests PRIVATE "${KAMITE_UNIT_ROOT}/src" "${CMAKE_CURRENT_LIST_DIR}")
target_include_directories(ghr_unit_tests PRIVATE "${KAMITE_UNIT_SINFL_DIR}")
target_compile_definitions(ghr_unit_tests PRIVATE KAMITE_SOURCE_DIR="${KAMITE_UNIT_ROOT}")
if(MSVC)
    target_compile_options(ghr_unit_tests PRIVATE /utf-8 /W3)
else()
    target_compile_options(ghr_unit_tests PRIVATE -Wall -Wextra)
endif()

add_test(NAME ghr_unit_tests COMMAND ghr_unit_tests)

# The file chooser portal's D-Bus side (src/kamite/portal_file_chooser.cpp, the Flatpak's picker): portal_pick_test asks
# through libdbus-1 as the launcher does, against a fake org.freedesktop.portal.Desktop on a private dbus-daemon
# (portal_bus.py, python3 + jeepney). Skipped (77) without dbus-daemon or jeepney; a failure under
# KAMITE_REQUIRE_ALL_TESTS=1 (the Nix check).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
    add_executable(portal_pick_test portal_pick_test.cpp "${KAMITE_UNIT_ROOT}/src/kamite/portal_file_chooser.cpp")
    target_include_directories(portal_pick_test PRIVATE "${KAMITE_UNIT_ROOT}/src")
    target_compile_options(portal_pick_test PRIVATE -Wall -Wextra)
    target_link_libraries(portal_pick_test PRIVATE ${CMAKE_DL_LIBS})
    target_link_libraries(ghr_unit_tests PRIVATE ${CMAKE_DL_LIBS})
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME portal_bus
            COMMAND "${Python3_EXECUTABLE}" -I "${CMAKE_CURRENT_LIST_DIR}/portal_bus.py" "$<TARGET_FILE:portal_pick_test>")
    else() # registered all the same: a skip, or a failure where every test must run
        add_test(NAME portal_bus COMMAND portal_pick_test missing-python)
    endif()
    set_tests_properties(portal_bus PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()

# The split game code's loader (src/kamite/game_library.cpp, the KAMITE_SPLIT_GAME_CODE prototype): game_library_test
# loads game_library_fixture.cpp built as one shared library per way a library can be refused, and once good. Runs in
# every Linux build (the loader is plain C++ and libdl), split or not.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
    set(KAMITE_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/game_library_fixtures")
    string(REPEAT "a" 64 kamiteIdA)
    string(REPEAT "b" 64 kamiteIdB)
    string(REPEAT "c" 64 kamiteIdC)
    set(fixture_good FIXTURE_ID="${kamiteIdA}")
    set(fixture_mismatch FIXTURE_ID="${kamiteIdB}")
    set(fixture_lying FIXTURE_ID="${kamiteIdC}" FIXTURE_SECTION_ID="${kamiteIdA}")
    set(fixture_small FIXTURE_ID="${kamiteIdA}" FIXTURE_SMALL_ENTRY=1)
    set(fixture_unresolved FIXTURE_ID="${kamiteIdA}" FIXTURE_UNRESOLVED=1)
    set(fixture_noentry FIXTURE_ID="${kamiteIdA}" FIXTURE_NO_ENTRY=1)
    set(fixture_nosection FIXTURE_ID="${kamiteIdA}" FIXTURE_NO_SECTION=1)
    set(KAMITE_FIXTURES fixture_good fixture_mismatch fixture_lying fixture_small fixture_unresolved fixture_noentry
        fixture_nosection)
    add_executable(game_library_test game_library_test.cpp "${KAMITE_UNIT_ROOT}/src/kamite/game_library.cpp")
    target_include_directories(game_library_test PRIVATE "${KAMITE_UNIT_ROOT}/src" "${CMAKE_CURRENT_LIST_DIR}")
    target_compile_definitions(game_library_test PRIVATE KAMITE_SPLIT_GAME_CODE=1)
    target_compile_options(game_library_test PRIVATE -Wall -Wextra)
    target_link_libraries(game_library_test PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(game_library_test PROPERTIES ENABLE_EXPORTS ON) # the fixtures call back into it
    include(CheckLinkerFlag)
    check_linker_flag(CXX "-Wl,-Bsymbolic-functions" KAMITE_FIXTURE_BSYMBOLIC_FUNCTIONS)
    check_linker_flag(CXX "-Wl,-z,lazy" KAMITE_FIXTURE_Z_LAZY)
    check_linker_flag(CXX "-Wl,-z,undefs" KAMITE_FIXTURE_Z_UNDEFS)
    foreach(fixtureName IN LISTS KAMITE_FIXTURES)
        set(fixtureDefinitions ${${fixtureName}})
        add_library(${fixtureName} MODULE game_library_fixture.cpp)
        target_include_directories(${fixtureName} PRIVATE "${KAMITE_UNIT_ROOT}/src" "${CMAKE_CURRENT_LIST_DIR}")
        target_compile_definitions(${fixtureName} PRIVATE ${fixtureDefinitions})
        set_target_properties(${fixtureName} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${KAMITE_FIXTURE_DIR}" PREFIX "lib")
        if(KAMITE_FIXTURE_BSYMBOLIC_FUNCTIONS) # as libghr_game.so, where the linker has it (src/CMakeLists.txt)
            target_link_options(${fixtureName} PRIVATE -Wl,-Bsymbolic-functions)
        endif()
        if(KAMITE_FIXTURE_Z_UNDEFS) # undefined until loaded, on purpose: undoes a distro's -z defs
            target_link_options(${fixtureName} PRIVATE -Wl,-z,undefs)
        endif()
        add_dependencies(game_library_test ${fixtureName})
    endforeach()
    # Lazy binding on the library itself, so only the loader's RTLD_NOW can refuse it at load (Nix's compiler wrapper
    # links with -z now by default, which would refuse it whatever the loader asked for).
    if(KAMITE_FIXTURE_Z_LAZY)
        target_link_options(fixture_unresolved PRIVATE -Wl,-z,lazy)
    endif()
    add_test(NAME game_library
        COMMAND game_library_test "${KAMITE_FIXTURE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/game_library_scratch")
    # Known gap with the portable toolchain (claude/split-game-library.md in the notes): zig links its C++ library
    # into each half, so the library's std::runtime_error is not the program's. That state is asserted exactly; the
    # day the gap closes the test fails and says so.
    if(KAMITE_LINUX_PORTABLE)
        set(kamiteExceptionMode exceptions-known-gap)
    else()
        set(kamiteExceptionMode exceptions)
    endif()
    add_test(NAME game_library_exceptions
        COMMAND game_library_test "${KAMITE_FIXTURE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/game_library_scratch_x"
                ${kamiteExceptionMode})
endif()
