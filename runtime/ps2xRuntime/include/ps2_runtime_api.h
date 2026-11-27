#pragma once

// The runtime's shared state across the split game code's boundary (the port's KAMITE_SPLIT_GAME_CODE: the generated
// code is a library, the runtime is in the program).
//
// A header-defined static (an inline variable, a static local of an inline function) is one object per module on
// Windows: the game library would keep its own scratchpad pointer, VU1 packet sink, log. With PS2X_SHARED_STATE such
// state is declared in the headers instead and defined once, in the program: src/lib/ps2_shared_state.cpp (the
// scratchpad pointer, the runtime log) and src/lib/vu/ps2_vu1_core.cpp (the VU state).
//
//   PS2X_RUNTIME_API  on the declaration of every VARIABLE the game code reads or writes from the runtime, and of the
//     accessor functions above. Windows: exported by the program, imported by the game library (PS2X_GAME_CODE); a DLL
//     can read data from the program only through __declspec(dllimport). Linux: default visibility.
//     Plain functions the game code calls do not need it: Windows routes those through the import library's thunks.
//     GHR.exe exports the runtime's and the IOP's symbols through a .def (src/CMakeLists.txt) and the game DLL links
//     GHR.lib. Both halves use the one shared C runtime (/MD, CMake's default here), which the log's std::deque relies
//     on: the game library's inline code fills it, the program owns it.
// Without PS2X_SHARED_STATE (every build but the split one) it is nothing and the headers keep their inline forms:
// the normal build's code is unchanged.
//
// Kept honest by two checks (src/CMakeLists.txt): tools/linux/check_shared_state.py (the headers' lines; every Linux
// build with tests) and tools/linux/check_shared_symbols.py (what the compiler emits from them; GCC builds with tests,
// the Nix check's included: clang has no -fkeep-inline-functions).

#if PS2X_SHARED_STATE
#if defined(_WIN32)
#if PS2X_GAME_CODE
#define PS2X_RUNTIME_API __declspec(dllimport)
#else
#define PS2X_RUNTIME_API __declspec(dllexport)
#endif
#else
#define PS2X_RUNTIME_API __attribute__((visibility("default")))
#endif
#else
#define PS2X_RUNTIME_API
#endif
