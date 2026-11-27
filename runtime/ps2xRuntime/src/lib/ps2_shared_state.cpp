// The runtime's shared state, defined once (ps2_runtime_api.h). Only with PS2X_SHARED_STATE (the split game code):
// the headers then declare these instead of defining them inline, so that the program and the game library see one
// scratchpad pointer and one log (the VU state: ps2_vu1_core.cpp). In every
// other build the headers keep their inline definitions and this file is empty.

#if PS2X_SHARED_STATE

#include "ps2_log.h"
#include "ps2_runtime_api.h"
#include "runtime/ps2_memory.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

PS2X_RUNTIME_API std::atomic<uint8_t *> g_ps2ScratchpadHostPtr{nullptr};

// The VU1 / VU0 state (packet sink, normalized-register shadows, VU0 scheduler flag) is defined in
// src/lib/vu/ps2_vu1_core.cpp, the one runtime file that includes ps2_vu1_impl.inl.

namespace ps2_log
{
    PS2X_RUNTIME_API std::mutex &runtime_log_mutex()
    {
        static std::mutex m;
        return m;
    }

    PS2X_RUNTIME_API std::deque<RuntimeLogEntry> &runtime_log_entries()
    {
        static std::deque<RuntimeLogEntry> entries;
        return entries;
    }

    PS2X_RUNTIME_API uint64_t &runtime_log_next_seq()
    {
        static uint64_t seq = 1;
        return seq;
    }

    PS2X_RUNTIME_API bool &runtime_log_paused()
    {
        static bool paused = false;
        return paused;
    }

#if AGRESSIVE_LOGS
    PS2X_RUNTIME_API std::string log_path()
    {
        static std::string path;
        if (path.empty())
            path = (std::filesystem::current_path() / "ps2_log.txt").string();
        return path;
    }

    PS2X_RUNTIME_API std::ostream &log_stream()
    {
        static std::ofstream f(log_path(), std::ios::out);
        return f.is_open() ? f : std::cerr;
    }

    PS2X_RUNTIME_API int &depth()
    {
        static thread_local int d = 0;
        return d;
    }
#endif
}

#endif
