// A stand-in game library for game_library_test.cpp (src/kamite/game_library.h), built once per FIXTURE_* variant by
// tests/unit/CMakeLists.txt: the good one, and one for each way a library can be refused.
//
//   FIXTURE_ID          the id its entry names (64 hex digits)
//   FIXTURE_SECTION_ID  the id in its .ghr_interface section (default: FIXTURE_ID)
//   FIXTURE_NO_SECTION  no .ghr_interface section
//   FIXTURE_NO_ENTRY    no ghr_game_library()
//   FIXTURE_SMALL_ENTRY its entry claims an older, smaller layout
//   FIXTURE_UNRESOLVED  needs a symbol the test program does not have

#include "kamite/game_library.h"
#include "game_library_shared.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// The test program's, found through its exports (-rdynamic) as the generated code finds the runtime's in GHR.
extern "C" void ghrTestHostNote(int what);
#if FIXTURE_UNRESOLVED
extern "C" void ghrTestNotInHost();
#endif

#ifndef FIXTURE_SECTION_ID
#define FIXTURE_SECTION_ID FIXTURE_ID
#endif

// A global with default visibility, as register_functions.cpp's g_ps2RecompiledFunctionTable is.
kamite::game_library::GameFunction g_ps2RecompiledFunctionTable[3] = {};

namespace
{
    using kamite::game_library::GameFunction;
    GameFunction *const g_table = g_ps2RecompiledFunctionTable;

    void functionA(uint8_t *, R5900Context *, PS2Runtime *) { ghrTestHostNote(1); }
    void functionB(uint8_t *, R5900Context *, PS2Runtime *)
    {
        ghrTestHostNote(2);
#if FIXTURE_UNRESOLVED
        ghrTestNotInHost();
#endif
    }


    // Filled by a static initializer, as register_functions.cpp's table is: it runs when the library is loaded, and
    // note 100 tells the test it did (a refused library must never get this far).
    struct Filler
    {
        Filler()
        {
            g_table[0] = functionA;
            g_table[2] = functionB;
            ghrTestHostNote(100);
        }
    } g_filler;

    const int g_packMarker = 0;
    const kamite::ModulePack *const g_packs[] = {reinterpret_cast<const kamite::ModulePack *>(&g_packMarker), nullptr};

    const kamite::ModulePack *const *packs(size_t &count)
    {
        count = 1;
        return g_packs;
    }

    void registerVu1() { ghrTestHostNote(3); }

#if !FIXTURE_NO_SECTION
    __attribute__((section(".ghr_interface"), used, retain)) const char kSection[] = "GHR-GAME-INTERFACE:" FIXTURE_SECTION_ID;
#endif
}

extern "C"
{
    // Calls slot `slot` through the library's own table: after the test program wrote that slot, this runs the test
    // program's function (one table, not a copy).
    __attribute__((visibility("default"))) void fixtureCallSlot(unsigned slot) { g_table[slot](nullptr, nullptr, nullptr); }
    __attribute__((visibility("default"))) int *fixtureSharedCounter() { return &sharedCounter(); }
    __attribute__((visibility("default"))) int *fixtureSharedInlineVariable() { return &sharedInlineVariable; }
    __attribute__((visibility("default"))) const void *fixturePacks() { return g_packs; }
    // The generated code throws std::runtime_error (the recompiler's preamble includes <stdexcept>) and the runtime
    // catches it by type: with the C++ library linked statically into each half (zig's libc++), that only works if
    // both halves agree on the exception's type.
    __attribute__((visibility("default"))) void fixtureThrow() { throw std::runtime_error("thrown in the library"); }
    // The other direction, which the game takes on every guest thread switch: GHR throws its own type
    // (EeDispatcherTransfer, EeScheduler.cpp) through generated-code frames, now the library's, and catches it itself.
    // The library's frame here has a destructor, which must run on the way through (note 9).
    __attribute__((visibility("default"))) void fixtureCallThrough(void (*hostFunction)())
    {
        struct Cleanup
        {
            ~Cleanup() { ghrTestHostNote(9); }
        } cleanup;
        hostFunction();
        ghrTestHostNote(10); // never reached: the host function throws
    }
    // Memory from one half freed by the other.
    __attribute__((visibility("default"))) std::string *fixtureNewString() { return new std::string(100, 'x'); }

#if !FIXTURE_NO_ENTRY
    __attribute__((visibility("default"))) const kamite::game_library::Entry *ghr_game_library()
    {
        static const kamite::game_library::Entry entry{
            FIXTURE_ID,
#if FIXTURE_SMALL_ENTRY
            8u,
#else
            static_cast<uint32_t>(sizeof(kamite::game_library::Entry)),
#endif
            kamite::game_library::kFlagStandIn,
            g_table,
            0x100000u,
            0x10000Cu,
            3u,
            &packs,
            &registerVu1,
        };
        return &entry;
    }
#endif
}
