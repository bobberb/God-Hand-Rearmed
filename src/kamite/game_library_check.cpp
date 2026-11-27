// The shared-state comparison (game_library.h checkSharedState): --check-game-library prints it, and every load runs it
// quietly. Split builds only.

#include "kamite/game_library.h"
#include "kamite/game_library_shared_state.h"

#include <cstdio>

namespace kamite::game_library
{
    int checkSharedState(const Result &loaded, bool print)
    {
        using Probe = size_t (*)(const void **, size_t);
        auto probe = loaded.handle ? reinterpret_cast<Probe>(findSymbol(loaded.handle, "ghr_game_shared_state")) : nullptr;
        if (!probe)
        {
            if (print)
                std::printf("shared state: not checked (the library has no probe: built before it had one)\n");
            return -1;
        }
        const void *mine[kSharedStateCount] = {};
        const void *theirs[kSharedStateCount] = {};
        sharedStateView(mine);
        if (probe(theirs, kSharedStateCount) != kSharedStateCount)
        {
            if (print)
                std::printf("shared state: the library's probe answered with another count\n");
            return 1;
        }
        size_t same = 0;
        for (size_t i = 0; i < kSharedStateCount; ++i)
        {
            const bool one = mine[i] == theirs[i];
            same += one;
            if (!one && print)
                std::printf("shared state: %s: TWO copies (GHR %p, the library %p)\n", kSharedStateNames[i], mine[i],
                            theirs[i]);
        }
        if (print)
            std::printf("shared state: %zu of %zu one object for both halves\n", same, kSharedStateCount);
        return same == kSharedStateCount ? 0 : 1;
    }
}
