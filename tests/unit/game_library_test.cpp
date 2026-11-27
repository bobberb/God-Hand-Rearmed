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

#include <algorithm>
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

    // A type of this half's own, as EeDispatcherTransfer is GHR's.
    struct HostTransfer
    {
        int value;
    };

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

namespace
{
    // A minimal PE32+ file as a Windows DLL's would start: the DOS header pointing at "PE\0\0", the file header, an
    // optional header of 240 bytes (PE32+ magic), the section table, then each section's raw data. Only what
    // readInterfaceId reads is filled; the rest is zero. For the PE half of the reader, tested here on Linux.
    struct PeSectionSpec
    {
        std::string name;    // up to 8 bytes
        std::string data;    // its raw bytes
        uint32_t virtualSize = 0; // 0: data's size
    };

    void put16(std::string &b, size_t at, uint16_t v) { b[at] = char(v & 0xFF); b[at + 1] = char(v >> 8); }
    void put32(std::string &b, size_t at, uint32_t v)
    {
        for (int i = 0; i < 4; ++i)
            b[at + i] = char((v >> (8 * i)) & 0xFF);
    }

    std::string makePe(const std::vector<PeSectionSpec> &sections, uint16_t machine = 0x8664, uint16_t characteristics = 0x2022,
                       uint16_t optionalMagic = 0x20B)
    {
        const size_t peOffset = 0x40, fileHeader = peOffset + 4, optional = fileHeader + 20, table = optional + 240;
        size_t raw = (table + 40 * sections.size() + 0x1FF) & ~size_t(0x1FF);
        std::string b(raw, '\0');
        b[0] = 'M';
        b[1] = 'Z';
        put32(b, 0x3C, uint32_t(peOffset));
        b.replace(peOffset, 4, std::string("PE\0\0", 4));
        put16(b, fileHeader, machine);
        put16(b, fileHeader + 2, uint16_t(sections.size()));
        put16(b, fileHeader + 16, 240);
        put16(b, fileHeader + 18, characteristics);
        put16(b, optional, optionalMagic);
        for (size_t i = 0; i < sections.size(); ++i)
        {
            const PeSectionSpec &sec = sections[i];
            const size_t entry = table + 40 * i;
            b.replace(entry, std::min<size_t>(8, sec.name.size()), sec.name.substr(0, 8));
            const size_t rawSize = (sec.data.size() + 0x1FF) & ~size_t(0x1FF);
            put32(b, entry + 8, sec.virtualSize ? sec.virtualSize : uint32_t(sec.data.size()));
            put32(b, entry + 16, uint32_t(rawSize));
            put32(b, entry + 20, uint32_t(b.size()));
            std::string block = sec.data;
            block.resize(rawSize, '\0');
            b += block;
        }
        return b;
    }

    std::string idSection(const std::string &id) { return std::string(gl::kSectionPrefix) + id + std::string(1, '\0'); }

    void checkPe(const fs::path &scratch)
    {
        // Each case reads first and checks after: check()'s arguments are evaluated in no set order, and the message
        // must carry this read's error, not the previous one's.
        auto refused = [&](const char *what, const std::string &name, const std::string &bytes, const std::string &because)
        {
            writeAll(scratch / name, bytes);
            std::string error;
            const std::string id = gl::readInterfaceId(scratch / name, error);
            check(id.empty() && contains(error, because), std::string(what) + ": " + (id.empty() ? error : "read id " + id));
        };
        {
            writeAll(scratch / "pe_good.dll", makePe({{".text", "code"}, {".ghrid", idSection(kA)}, {".rdata", "data"}}));
            std::string error;
            const std::string id = gl::readInterfaceId(scratch / "pe_good.dll", error);
            check(id == kA, "PE: the id from the .ghrid section: " + error);
        }
        refused("PE without the section", "pe_none.dll", makePe({{".text", "code"}}), "no .ghrid section");
        refused("PE with two id sections", "pe_two.dll", makePe({{".ghrid", idSection(kA)}, {".ghrid", idSection(kB)}}),
                "two .ghrid");
        std::string bad = idSection(kA);
        bad[std::strlen(gl::kSectionPrefix) + 3] = 'Z';
        refused("PE id not lowercase hex", "pe_hex.dll", makePe({{".ghrid", bad}}), "malformed");
        std::string badPrefix = idSection(kA);
        badPrefix[0] = 'X';
        refused("PE id with another prefix", "pe_prefix.dll", makePe({{".ghrid", badPrefix}}), "malformed");
        refused("PE id section too short", "pe_short.dll", makePe({{".ghrid", std::string(gl::kSectionPrefix) + "abc"}}),
                "malformed");
        refused("PE id section whose virtual size is short", "pe_vsize.dll", makePe({{".ghrid", idSection(kA), 10}}),
                "malformed");
        refused("PE for another machine", "pe_i386.dll", makePe({{".ghrid", idSection(kA)}}, 0x14C), "64-bit Windows DLL");
        refused("PE that is not a DLL", "pe_exe.dll", makePe({{".ghrid", idSection(kA)}}, 0x8664, 0x0022),
                "64-bit Windows DLL");
        refused("PE32 (not PE32+)", "pe_pe32.dll", makePe({{".ghrid", idSection(kA)}}, 0x8664, 0x2022, 0x10B),
                "64-bit Windows DLL");
        const std::string good = makePe({{".ghrid", idSection(kA)}});
        refused("PE cut in its header", "pe_cut.dll", good.substr(0, 0x50), "not a PE");
        refused("PE cut in its sections", "pe_cut2.dll", good.substr(0, 0x1F0), "past its end");
        std::string far = good;
        put32(far, 0x3C, 0x7FFFFFF0u); // e_lfanew far past the end
        refused("PE header pointer past the end", "pe_far.dll", far, "not a PE");
        const gl::Result r = gl::load({scratch / "pe_good.dll"}, kB);
        check(r.status == gl::Status::Mismatch, "a PE built for another GHR is refused before loading: " + r.message);
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

    // The PE reader (Windows' file format), from files built here.
    checkPe(scratch);

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
        check(text.status == gl::Status::NotAGameLibrary && contains(text.message, "neither an ELF nor a PE"),
              "text file: " + text.message);

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

        // This half's exception through the library's frames, caught here by its type, the library's cleanup run.
        auto callThrough = symbol<void (*)(void (*)())>(fixture("fixture_good"), "fixtureCallThrough");
        check(callThrough != nullptr, "good: fixtureCallThrough found");
        if (callThrough)
        {
            bool caughtTransfer = false;
            try
            {
                callThrough([] { throw HostTransfer{42}; });
            }
            catch (const HostTransfer &transfer)
            {
                caughtTransfer = transfer.value == 42;
            }
            catch (...)
            {
            }
            check(caughtTransfer, "good: the program's exception crosses the library's frames and is caught by type here");
            check(notesOf(9) == 1 && notesOf(10) == 0, "good: the library's destructor ran on the way through");
        }

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
