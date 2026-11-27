#pragma once

// The split game code (prototype, -DKAMITE_SPLIT_GAME_CODE=ON; Linux, and Windows with MSVC): the recompiled game code
// (the main program, the 81 room modules, the VU1 microcode) is a shared library of its own (libghr_game.so on Linux,
// ghr_game.dll on Windows), which GHR loads at start
// instead of linking it in. GHR, the launcher and the runtime then build and ship with no game code in them, and the
// library is built on the player's machine from their own disc (GHR-Tech-Deck issue #1).
//
// The two halves meet in one place: the library exports ghr_game_library(), which returns an Entry. Everything else
// the generated code needs (the runtime's functions and globals) it takes from GHR itself, which is linked with
// ENABLE_EXPORTS (-rdynamic) so the library's undefined symbols resolve against the program that loads it. On Windows
// GHR.exe exports the runtime through a .def file and the DLL links GHR.lib, its import library; a DLL never shares a
// header-defined static with the program there, which is what ps2_runtime_api.h's shared state is for. On Linux it also
// keeps one copy of every inline variable and function-local static the runtime headers define (the library's
// references bind to GHR's), which is why the library is not linked with -Bsymbolic or hidden visibility. It is linked
// with -Bsymbolic-functions: its own functions bind inside it (game-to-game calls skip the PLT), its data does not.
// For the same reason GHR links the runtime as whole archives: the generated code calls runtime functions nothing in
// GHR itself calls (ps2ee::div), which a plain static link would leave out.
//
// The interface id: both halves compile against the same runtime headers, struct layouts, macros and compiler, or the
// library's code is wrong for this GHR. src/CMakeLists.txt hashes all of that (SHA-256) into
// kamite_generated/kamite/game_interface.h at configure time. The library carries the id twice: in a section of its
// file (ELF .ghr_interface, PE .ghrid: a PE section name has 8 bytes at most), which load() reads from the file before
// running any of the library's code, and in its Entry.
// A different id = built for another GHR: the library is refused, and the player builds it again.
//
// This header is included by both halves and by tests/unit/game_library_test.cpp, so it names the runtime's types
// without including the runtime.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct R5900Context;
class PS2Runtime;

namespace kamite
{
    struct ModulePack; // module_registry.h
}

namespace kamite::game_library
{
    // PS2Runtime::RecompiledFunction (game_library_entry.cpp checks they are the same type).
    using GameFunction = void (*)(uint8_t *, R5900Context *, PS2Runtime *);

#if defined(_WIN32)
    constexpr const char *kFileName = "ghr_game.dll";
#else
    constexpr const char *kFileName = "libghr_game.so";
#endif
    constexpr const char *kEntrySymbol = "ghr_game_library";
    constexpr const char *kElfSectionName = ".ghr_interface";
    constexpr const char *kPeSectionName = ".ghrid"; // PE section names: 8 bytes at most
    constexpr const char *kSectionPrefix = "GHR-GAME-INTERFACE:"; // then the id (64 hex digits), then NUL
    constexpr size_t kIdLength = 64;

    constexpr uint32_t kFlagStandIn = 1u; // built on tools/linux/stub_game_code.py's stand-in: cannot play the game

    // What ghr_game_library() returns. interfaceId and size come first and never move, so a GHR can read them from
    // a library built for another layout; every other field is read only once interfaceId has matched.
    struct Entry
    {
        const char *interfaceId; // the library's id, 64 hex digits
        uint32_t size;           // sizeof(Entry) where the library was built
        uint32_t flags;          // kFlag*
        // The dense function table (register_functions.cpp's g_ps2RecompiledFunctionTable and its bounds): the
        // library's own array. GHR reads and writes it in place (natives, hooks), so both halves see one table.
        GameFunction *functions;
        uint32_t base;
        uint32_t end;
        uint32_t slotCount;
        // The room modules' packs (src/CMakeLists.txt's module index, compiled into the library).
        const ModulePack *const *(*modulePacks)(size_t &count);
        // generated/vu1's registerGeneratedVu1Programs; null when the build has no VU1 code.
        void (*registerVu1Programs)();
    };

    enum class Status
    {
        Loaded,
        Missing,         // no library in any of the places looked
        NotAGameLibrary, // not an ELF shared library with a .ghr_interface section, or no ghr_game_library()
        Mismatch,        // built for another GHR (interface id differs)
        LoadFailed,      // dlopen refused it (e.g. a symbol this GHR does not provide)
    };

    struct Result
    {
        Status status = Status::Missing;
        std::filesystem::path path; // the library tried ("" when Missing)
        std::string message;        // for the log and the launcher, "" when Loaded
        const Entry *entry = nullptr; // Loaded only; valid for the rest of the process (never unloaded)
        void *handle = nullptr;       // Loaded only: the dlopen handle (for --check-game-library's probes)
    };

    const char *statusName(Status status);

    // Where GHR looks, in order: $KAMITE_GAME_LIBRARY alone when it is set and non-empty; else the play folder's
    // game_code/ (where a player's build goes: writable in every package), then the program folder (a developer's
    // build puts it next to GHR).
    std::vector<std::filesystem::path> candidates(const char *environment, const std::filesystem::path &home,
                                                  const std::filesystem::path &programDir);

    // The interface id in a file's .ghr_interface section, read from the file without loading it. "" (and why, in
    // `error`) when the file is not a 64-bit little-endian ELF or has no well-formed section.
    // Either format, whatever the platform (the tests read both on Linux).
    std::string readInterfaceId(const std::filesystem::path &file, std::string &error);

    // Loads the first candidate that exists and checks it against `expectedId`. Nothing of the library runs before
    // its section's id has matched. A refused library is unloaded again.
    Result load(const std::vector<std::filesystem::path> &candidates, const std::string &expectedId);

#if KAMITE_SPLIT_GAME_CODE
    // Makes `entry` the library behind kamite::modulePacks() (function_hooks.cpp) and registerVu1Programs().
    void use(const Entry *entry);
    // Registers the library's VU1 programs; false (and nothing done) when it has none.
    bool registerVu1Programs();
    // A symbol the loaded library exports (dlsym / GetProcAddress); null when it has none.
    void *findSymbol(void *handle, const char *name);
    // --check-game-library: compares the runtime state the library sees with GHR's (game_library_shared_state.h),
    // printing a line per object. 0 = one object each, 1 = at least one differs, -1 = the library has no probe
    // (ghr_game_shared_state, in every library built since it existed). game_library_check.cpp.
    int checkSharedState(const Result &loaded);
#endif
}
