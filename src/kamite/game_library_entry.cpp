// The game library's one export (game_library.h): compiled into libghr_game.so / ghr_game.dll with the generated code,
// never into GHR. Built only with -DKAMITE_SPLIT_GAME_CODE=ON (src/CMakeLists.txt).

#include "kamite/game_library.h"
#include "kamite/game_interface.h" // KAMITE_GAME_INTERFACE: the id src/CMakeLists.txt computed
#include "kamite/module_registry.h"

#include "ps2_runtime.h"

#include <type_traits>

#if defined(_MSC_VER)
#define GHR_GAME_EXPORT extern "C" __declspec(dllexport)
#else
#define GHR_GAME_EXPORT extern "C" __attribute__((visibility("default")))
#endif

static_assert(std::is_same_v<kamite::game_library::GameFunction, PS2Runtime::RecompiledFunction>,
              "game_library.h's GameFunction must be the runtime's RecompiledFunction");

namespace kamite
{
    // The module index src/CMakeLists.txt writes for the split build (kamite_generated/kamite/module_index.cpp, the
    // same list under another name: GHR's own modulePacks forwards to it, and one name for both would bind the
    // library's address of it to GHR's).
    const ModulePack *const *gameLibraryModulePacks(size_t &count);
}

#if KAMITE_GAME_HAS_VU1
void registerGeneratedVu1Programs(); // generated/vu1/vu1_programs.cpp
#endif

namespace
{
    // Read from the file by GHR before the library is loaded (game_library.cpp readInterfaceId). `retain` and the
    // entry's pointer into it both keep it: a linker that collects unreferenced sections (zig's lld does) dropped it
    // with `used` alone. The entry's id is these same bytes, so the two can never disagree. A PE section's name has 8
    // bytes at most: .ghrid there.
#if defined(_MSC_VER)
#pragma section(".ghrid", read)
    __declspec(allocate(".ghrid")) const char kInterfaceSection[] =
#else
    __attribute__((section(".ghr_interface"), used, retain)) const char kInterfaceSection[] =
#endif
        "GHR-GAME-INTERFACE:" KAMITE_GAME_INTERFACE;
    constexpr size_t kPrefixLength = sizeof("GHR-GAME-INTERFACE:") - 1;
    static_assert(sizeof kInterfaceSection == kPrefixLength + kamite::game_library::kIdLength + 1,
                  "KAMITE_GAME_INTERFACE must be 64 hex digits");

    constexpr uint32_t kFlags =
#if KAMITE_GAME_STAND_IN
        kamite::game_library::kFlagStandIn;
#else
        0u;
#endif
}

#if KAMITE_GAME_STAND_IN
#include "ps2_runtime_macros.h"
// The stand-in has no game code, so its library needs almost nothing from GHR and would load even if GHR lacked runtime
// code the real game code calls. This does what recompiled FPU code does (ps2EeDiv / ps2EeSqrt fall back to
// ps2ee::div / ps2ee::sqrt, which nothing in GHR itself calls): the stand-in's library loads only if GHR carries them.
GHR_GAME_EXPORT uint32_t ghr_stand_in_probe(uint32_t a, uint32_t b)
{
    return ps2EeDiv(a, b) ^ ps2EeSqrt(a);
}
#endif

#include "kamite/game_library_shared_state.h"
// The runtime state the game code shares with GHR, as this library sees it (game_library_shared_state.h):
// GHR --check-game-library compares it with its own. In every game library, the real one too.
GHR_GAME_EXPORT size_t ghr_game_shared_state(const void **out, size_t count)
{
    if (count < kamite::game_library::kSharedStateCount)
        return 0;
    kamite::game_library::sharedStateView(out);
    return kamite::game_library::kSharedStateCount;
}

GHR_GAME_EXPORT const kamite::game_library::Entry *ghr_game_library()
{
    // Built at the first call, after dlopen has run the generated code's initializers (register_functions.cpp's
    // table filler among them).
    static const kamite::game_library::Entry entry{
        kInterfaceSection + kPrefixLength,
        static_cast<uint32_t>(sizeof(kamite::game_library::Entry)),
        kFlags,
        g_ps2RecompiledFunctionTable,
        g_ps2RecompiledFunctionTableBase,
        g_ps2RecompiledFunctionTableEnd,
        g_ps2RecompiledFunctionTableSlotCount,
        &kamite::gameLibraryModulePacks,
#if KAMITE_GAME_HAS_VU1
        &registerGeneratedVu1Programs,
#else
        nullptr,
#endif
    };
    return &entry;
}
