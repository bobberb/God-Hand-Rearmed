#pragma once

// The runtime's shared state across the split game code's boundary (the port's KAMITE_SPLIT_GAME_CODE: the generated
// code is a library, the runtime is in the program).
//
// A header-defined static (an inline variable, a static local of an inline function) is one object per module on
// Windows: the game library would keep its own scratchpad pointer, VU1 packet sink, log. With PS2X_SHARED_STATE such
// state is declared here-annotated and defined once, in src/lib/ps2_shared_state.cpp, which is in the program:
//   PS2X_RUNTIME_API  on the declaration of every function or variable the game code takes from the runtime.
//     Windows: exported by the program, imported by the game library (PS2X_GAME_CODE). Linux: default visibility.
// Without PS2X_SHARED_STATE (every build but the split one) it is nothing and the headers keep their inline forms:
// the normal build's code is unchanged. tools/linux/check_shared_state.py keeps new header statics out.

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
