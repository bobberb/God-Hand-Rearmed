// The split game code's loader (src/kamite/game_library.cpp) against real shared libraries: game_library_fixture.cpp
// built once per way a library can be refused, and once good.
//
//   game_library_test <fixture folder> <scratch folder> [exceptions | exceptions-known-gap]
//
// "exceptions": only one check, that a C++ exception thrown in the library is caught by its type in the program.
// "exceptions-known-gap": the portable toolchain's state, exactly: the library loads, its exception reaches the program
// (catch (...)) but not as its type, because zig links its libc++ and libc++abi, hidden, into each half, so each has
// its own std::runtime_error. Anything else fails it, the gap closing included (then switch the test to "exceptions").
//
// Exit 0 = every check passed; otherwise each failure is printed and the exit code is 1.

#include "kamite/game_library.h"
#include "game_library_shared.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace gl = kamite::game_library;

namespace kamite
{
    const ModulePack *const *modulePacks(size_t &count); // game_library.cpp's forwarder
}

namespace
{
    std::vector<int> g_notes;
    int g_failures = 0;
    int g_checks = 0;

    void check(bool ok, const std::string &what)
    {
        ++g_checks;
        if (!ok)
        {
            ++g_failures;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    bool contains(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }

    int notesOf(int what)
    {
        int n = 0;
        for (int note : g_notes)
            n += note == what;
        return n;
    }

    const std::string kA(64, 'a');
    const std::string kB(64, 'b');

    void hostFunction(uint8_t *, R5900Context *, PS2Runtime *) { g_notes.push_back(7); }

    std::string readAll(const fs::path &file)
    {
        std::ifstream in(file, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    }

    void writeAll(const fs::path &file, const std::string &bytes)
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    template <typename T>
    T symbol(const fs::path &library, const char *name)
    {
        // RTLD_NOLOAD: the library load() already opened (the same handle), never a second copy.
        void *handle = dlopen(library.c_str(), RTLD_NOW | RTLD_NOLOAD);
        return handle ? reinterpret_cast<T>(dlsym(handle, name)) : nullptr;
    }
}

extern "C" __attribute__((visibility("default"))) void ghrTestHostNote(int what)
{
    g_notes.push_back(what);
}

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4)
    {
        std::fprintf(stderr, "usage: game_library_test <fixture folder> <scratch folder> [exceptions | exceptions-known-gap]\n");
        return 2;
    }
    const fs::path fixtures = argv[1];
    const fs::path scratch = argv[2];
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    auto fixture = [&](const char *name) { return fixtures / (std::string("lib") + name + ".so"); };

    const std::string mode = argc == 4 ? argv[3] : "";
    if (mode == "exceptions" || mode == "exceptions-known-gap")
    {
        const gl::Result good = gl::load({fixture("fixture_good")}, kA);
        check(good.status == gl::Status::Loaded, "exceptions: the good fixture loads: " + good.message);
        auto throwing = symbol<void (*)()>(fixture("fixture_good"), "fixtureThrow");
        check(throwing != nullptr, "exceptions: fixtureThrow found");
        std::string caught = "nothing";
        try
        {
            if (throwing)
                throwing();
        }
        catch (const std::runtime_error &e)
        {
            caught = e.what();
        }
        catch (...)
        {
            caught = "caught, but not as std::runtime_error";
        }
        if (mode == "exceptions")
            check(caught == "thrown in the library", "the library's std::runtime_error is caught by type here: " + caught);
        else
            check(caught == "caught, but not as std::runtime_error",
                  "known gap: the library's exception arrives, but not as std::runtime_error (now: " + caught + ")");
        std::printf("%d checks, %d failed\n", g_checks, g_failures);
        return g_failures ? 1 : 0;
    }

    // Where GHR looks.
    {
        const auto env = gl::candidates("/x/lib.so", "/home", "/prog");
        check(env.size() == 1 && env[0] == "/x/lib.so", "candidates: $KAMITE_GAME_LIBRARY alone");
        const auto none = gl::candidates("", "/home", "/prog");
        check(none.size() == 2 && none[0] == fs::path("/home/game_code/libghr_game.so") &&
                  none[1] == fs::path("/prog/libghr_game.so"),
              "candidates: the play folder's game_code/, then the program folder");
        check(gl::candidates(nullptr, "/home", "/prog") == none, "candidates: unset is the same as empty");
        const auto relative = gl::candidates("some/lib.so", "/home", "/prog");
        check(relative.size() == 1 && relative[0].is_absolute() && relative[0].filename() == "lib.so",
              "candidates: a relative $KAMITE_GAME_LIBRARY is made absolute (dlopen would search for it otherwise)");
    }

    // Missing.
    {
        const gl::Result r = gl::load({scratch / "nope.so", scratch / "nope2.so"}, kA);
        check(r.status == gl::Status::Missing && !r.entry, "missing: status");
        check(contains(r.message, "not built yet") && contains(r.message, "nope2.so"), "missing: message names the places: " + r.message);
    }

    // Not a library at all, and a truncated one.
    {
        writeAll(scratch / "text.so", "this is not an ELF file, but it is long enough to hold an ELF header.......");
        const gl::Result text = gl::load({scratch / "text.so"}, kA);
        check(text.status == gl::Status::NotAGameLibrary && contains(text.message, "not an ELF"), "text file: " + text.message);

        writeAll(scratch / "short.so", readAll(fixture("fixture_good")).substr(0, 200));
        const gl::Result shortFile = gl::load({scratch / "short.so"}, kA);
        check(shortFile.status == gl::Status::NotAGameLibrary && contains(shortFile.message, "past its end"),
              "truncated library: " + shortFile.message);
    }

    // A section id that is not 64 lowercase hex digits.
    {
        std::string bytes = readAll(fixture("fixture_good"));
        const size_t at = bytes.find(std::string(gl::kSectionPrefix) + kA);
        check(at != std::string::npos, "the good fixture carries its section's text");
        if (at != std::string::npos)
        {
            bytes[at + std::strlen(gl::kSectionPrefix) + 10] = 'G';
            writeAll(scratch / "corrupt.so", bytes);
            // The id first: check()'s arguments are evaluated in no set order, and the message reads `error`.
            std::string error;
            const std::string corruptId = gl::readInterfaceId(scratch / "corrupt.so", error);
            check(corruptId.empty() && contains(error, "malformed"), "corrupt section id: " + error);
            const gl::Result r = gl::load({scratch / "corrupt.so"}, kA);
            check(r.status == gl::Status::NotAGameLibrary, "corrupt section id: refused");

            bytes = readAll(fixture("fixture_good"));
            bytes[at] = 'X'; // "XHR-GAME-INTERFACE:", the same length
            writeAll(scratch / "prefix.so", bytes);
            const std::string prefixId = gl::readInterfaceId(scratch / "prefix.so", error);
            check(prefixId.empty() && contains(error, "malformed"), "wrong section prefix: " + error);
        }
    }

    // The refusals. None of them may run the library's code: its initializer's note 100 must not appear, except for
    // the two that are only found out after dlopen (the entry's own id, its size), which then unload it.
    {
        const gl::Result r = gl::load({fixture("fixture_nosection")}, kA);
        check(r.status == gl::Status::NotAGameLibrary && contains(r.message, ".ghr_interface"), "no section: " + r.message);
    }
    {
        const gl::Result r = gl::load({fixture("fixture_mismatch")}, kA);
        check(r.status == gl::Status::Mismatch, "mismatch: status");
        check(contains(r.message, std::string(12, 'b')) && contains(r.message, std::string(12, 'a')),
              "mismatch: message names both ids: " + r.message);
    }
    {
        const gl::Result r = gl::load({fixture("fixture_unresolved")}, kA);
        check(r.status == gl::Status::LoadFailed && contains(r.message, "ghrTestNotInHost"),
              "unresolved symbol: refused at load, named: " + r.message);
    }
    check(notesOf(100) == 0, "no refused library's code ran before its id matched");
    {
        const gl::Result r = gl::load({fixture("fixture_noentry")}, kA);
        check(r.status == gl::Status::NotAGameLibrary && contains(r.message, "ghr_game_library"), "no entry: " + r.message);
    }
    {
        const gl::Result r = gl::load({fixture("fixture_lying")}, kA);
        check(r.status == gl::Status::Mismatch && contains(r.message, "entry names interface"), "entry id differs: " + r.message);
        // dlclose'd, but whether it really leaves memory is the C library's call (GCC's unique symbols, which inline
        // statics use, pin a library): what matters is that GHR never gets its entry.
        check(!r.entry, "entry id differs: no entry handed out");
    }
    {
        const gl::Result r = gl::load({fixture("fixture_small")}, kA);
        check(r.status == gl::Status::Mismatch && contains(r.message, "older entry layout"), "small entry: " + r.message);
    }

    // The first library that exists is the one used: a stale one in the play folder is reported, not skipped for an
    // older one in the program folder.
    {
        const gl::Result r = gl::load({scratch / "absent.so", fixture("fixture_mismatch"), fixture("fixture_good")}, kA);
        check(r.status == gl::Status::Mismatch && r.path == fixture("fixture_mismatch"), "first existing candidate wins");
    }

    // Before a library is in use, no modules.
    {
        size_t count = 99;
        const auto *packs = kamite::modulePacks(count);
        check(count == 0 && packs && !packs[0], "modulePacks before use(): none");
        check(!gl::registerVu1Programs(), "registerVu1Programs before use(): nothing");
    }

    // Loaded.
    std::string error;
    check(gl::readInterfaceId(fixture("fixture_good"), error) == kA, "the good fixture's id, read from the file");
    const int initializersBefore = notesOf(100);
    const gl::Result good = gl::load({scratch / "absent.so", fixture("fixture_good")}, kA);
    check(good.status == gl::Status::Loaded && good.entry && good.message.empty(), "good: loaded: " + good.message);
    if (good.entry)
    {
        const gl::Entry &e = *good.entry;
        check(good.path == fixture("fixture_good"), "good: the path of the one loaded");
        check(notesOf(100) == initializersBefore + 1, "good: its static initializer ran once");
        check(e.base == 0x100000u && e.end == 0x10000Cu && e.slotCount == 3u, "good: the table's bounds");
        check(e.flags & gl::kFlagStandIn, "good: the stand-in flag");
        check(e.functions[0] && !e.functions[1] && e.functions[2], "good: the table as its initializer filled it");

        e.functions[0](nullptr, nullptr, nullptr);
        check(notesOf(1) == 1, "good: a library function runs and calls back into the program");

        // One table: GHR's write (a native, a hook) is what the library's own dispatch sees.
        e.functions[1] = &hostFunction;
        auto callSlot = symbol<void (*)(unsigned)>(fixture("fixture_good"), "fixtureCallSlot");
        check(callSlot != nullptr, "good: fixtureCallSlot found");
        if (callSlot)
        {
            callSlot(1);
            check(notesOf(7) == 1, "good: the program's write to a slot is seen by the library (one table, not a copy)");
        }

        // One object for header-defined statics and inline variables.
        auto counter = symbol<int *(*)()>(fixture("fixture_good"), "fixtureSharedCounter");
        auto inlineVariable = symbol<int *(*)()>(fixture("fixture_good"), "fixtureSharedInlineVariable");
        check(counter && counter() == &sharedCounter(), "good: an inline function's static is one object in both");
        check(inlineVariable && inlineVariable() == &sharedInlineVariable, "good: an inline variable is one object in both");

        // A std::string made by the library, grown and freed by this half.
        auto newString = symbol<std::string *(*)()>(fixture("fixture_good"), "fixtureNewString");
        check(newString != nullptr, "good: fixtureNewString found");
        if (newString)
        {
            std::string *text = newString();
            check(text->size() == 100 && (*text)[99] == 'x', "good: a std::string made in the library reads here");
            text->append(1000, 'y');
            check(text->size() == 1100, "good: and grows here");
            delete text;
        }

        // The forwarders.
        gl::use(good.entry);
        size_t count = 0;
        const auto *packs = kamite::modulePacks(count);
        auto fixturePacks = symbol<const void *(*)()>(fixture("fixture_good"), "fixturePacks");
        check(count == 1 && fixturePacks && static_cast<const void *>(packs) == fixturePacks(),
              "good: modulePacks() hands out the library's packs");
        check(gl::registerVu1Programs() && notesOf(3) == 1, "good: registerVu1Programs() runs the library's");
    }

    std::printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
