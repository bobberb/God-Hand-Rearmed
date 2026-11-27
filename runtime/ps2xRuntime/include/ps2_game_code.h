#pragma once

// Where the runtime (and the port's samplers) find the recompiled game code's dense function table.
//
// Normal build: the game code is linked into the program, and these name register_functions.cpp's globals: the very
// tokens the code used before this header existed, so the machine code is the same (macros, not inline functions:
// even an inlined accessor moved GCC's inlining and register choices in ps2_runtime.cpp's dispatch path).
// Split build (PS2X_SPLIT_GAME_CODE=1, the port's KAMITE_SPLIT_GAME_CODE prototype): the game code is a shared library
// the program loads at start, and the program has no such globals. The loader (src/kamite/game_library.h) copies the
// library's table pointer and bounds into g_gameTable before anything reads the table; until then slotCount is 0, so
// every address is outside the table (lookups find nothing, replaceFunction refuses).

#include "ps2_runtime.h"

#include <cstdint>

#if PS2X_SPLIT_GAME_CODE
namespace ps2x
{
    struct GameTable
    {
        PS2Runtime::RecompiledFunction *functions = nullptr;
        uint32_t base = 0u;
        uint32_t end = 0u;
        uint32_t slotCount = 0u;
    };
    extern GameTable g_gameTable; // shared-state: the program's only (ps2_runtime.cpp); the game code never reads it
}
#define PS2X_GAME_TABLE (::ps2x::g_gameTable.functions)
#define PS2X_GAME_TABLE_BASE (::ps2x::g_gameTable.base)
#define PS2X_GAME_TABLE_END (::ps2x::g_gameTable.end)
#define PS2X_GAME_TABLE_SLOT_COUNT (::ps2x::g_gameTable.slotCount)
#else
#define PS2X_GAME_TABLE g_ps2RecompiledFunctionTable
#define PS2X_GAME_TABLE_BASE g_ps2RecompiledFunctionTableBase
#define PS2X_GAME_TABLE_END g_ps2RecompiledFunctionTableEnd
#define PS2X_GAME_TABLE_SLOT_COUNT g_ps2RecompiledFunctionTableSlotCount
#endif
