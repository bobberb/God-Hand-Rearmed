#pragma once

// The runtime state the game code shares with GHR (ps2_runtime_api.h), as the module that compiles this header sees
// it: the address of each object. Compiled into GHR (game_library_check.cpp) and into every game library
// (game_library_entry.cpp, ghr_game_shared_state); `GHR --check-game-library` compares the two lists. Equal
// addresses = one object for both halves.
//
// On Windows a header-defined static would give the library its own copy. The split build's test library linked with
// -Bsymbolic (src/CMakeLists.txt, ghr_game_local) binds the same way on Linux for ordinary data, so one of these put
// back in a header shows here as two addresses. Not for thread_local data: -Bsymbolic leaves it one object with the
// program, so the packet sink (a thread_local) reverted to its header form would pass here. That one, and every other
// piece of header state, is the compiler check's (tools/linux/check_shared_symbols.py), which lists it.
//
// Split builds only (PS2X_SHARED_STATE). Includes ps2_vu1_impl.inl, as the compiled VU1 code does.

#include "ps2_log.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_impl.inl"

#include <cstddef>

namespace kamite::game_library
{
    inline constexpr size_t kSharedStateCount = 9;
    inline constexpr const char *kSharedStateNames[kSharedStateCount] = {
        "scratchpad host pointer",
        "runtime log entries",
        "runtime log mutex",
        "runtime log sequence number",
        "runtime log paused flag",
        "VU1 packet sink (this thread; thread_local: not caught by the -Bsymbolic test)",
        "VU1 normalized shadow",
        "VU0 normalized shadow",
        "VU0 scheduler clean flag",
    };

    inline void sharedStateView(const void *out[kSharedStateCount])
    {
        out[0] = &ps2ScratchpadHostPtrStorage();
        out[1] = &ps2_log::runtime_log_entries();
        out[2] = &ps2_log::runtime_log_mutex();
        out[3] = &ps2_log::runtime_log_next_seq();
        out[4] = &ps2_log::runtime_log_paused();
        out[5] = &(PS2X_VU1_PACKET_SINK);
        out[6] = &vu1NormShadow;
        out[7] = &vu0NormShadow;
        out[8] = &vu0SchedulerClean;
    }
}
