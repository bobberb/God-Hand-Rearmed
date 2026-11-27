// The split game code's loader (game_library.h).

#include "kamite/game_library.h"

#include <dlfcn.h>
#include <elf.h>

#include <cstring>
#include <fstream>
#include <limits>
#include <system_error>

namespace fs = std::filesystem;

namespace kamite::game_library
{
    namespace
    {
        // Read `size` bytes at `offset`, false when the file is shorter (or the range wraps).
        bool readAt(std::ifstream &in, uint64_t fileSize, uint64_t offset, void *out, uint64_t size)
        {
            if (offset > fileSize || size > fileSize - offset)
                return false;
            in.clear();
            in.seekg(static_cast<std::streamoff>(offset));
            in.read(static_cast<char *>(out), static_cast<std::streamsize>(size));
            return static_cast<uint64_t>(in.gcount()) == size;
        }

        bool lowerHex(const char *text, size_t length)
        {
            for (size_t i = 0; i < length; ++i)
                if (!((text[i] >= '0' && text[i] <= '9') || (text[i] >= 'a' && text[i] <= 'f')))
                    return false;
            return true;
        }

        std::string shortId(const std::string &id)
        {
            return id.size() > 12 ? id.substr(0, 12) : id;
        }

        std::string list(const std::vector<fs::path> &paths)
        {
            std::string out;
            for (const fs::path &p : paths)
                out += (out.empty() ? "" : ", ") + p.string();
            return out;
        }

        Result refuse(Status status, const fs::path &path, std::string message)
        {
            Result r;
            r.status = status;
            r.path = path;
            r.message = std::move(message);
            return r;
        }
    }

    const char *statusName(Status status)
    {
        switch (status)
        {
        case Status::Loaded:
            return "loaded";
        case Status::Missing:
            return "missing";
        case Status::NotAGameLibrary:
            return "not a game library";
        case Status::Mismatch:
            return "built for another GHR";
        case Status::LoadFailed:
            return "load failed";
        }
        return "?";
    }

    std::vector<fs::path> candidates(const char *environment, const fs::path &home, const fs::path &programDir)
    {
        if (environment && *environment)
        {
            // Absolute: dlopen searches the library paths for a name without a slash, so a relative name could be
            // checked here as one file and loaded as another.
            std::error_code ec;
            const fs::path absolute = fs::absolute(environment, ec);
            return {ec ? fs::path(environment) : absolute};
        }
        return {home / "game_code" / kFileName, programDir / kFileName};
    }

    std::string readInterfaceId(const fs::path &file, std::string &error)
    {
        std::ifstream in(file, std::ios::binary);
        if (!in)
        {
            error = "cannot be opened";
            return "";
        }
        in.seekg(0, std::ios::end);
        const std::streamoff end = in.tellg();
        if (end < 0)
        {
            error = "cannot be read";
            return "";
        }
        const uint64_t fileSize = static_cast<uint64_t>(end);

        Elf64_Ehdr header{};
        if (!readAt(in, fileSize, 0, &header, sizeof header) || std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0)
        {
            error = "is not an ELF file";
            return "";
        }
        if (header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB)
        {
            error = "is not a 64-bit little-endian ELF file";
            return "";
        }
        if (header.e_type != ET_DYN)
        {
            error = "is not a shared library";
            return "";
        }
        if (header.e_shoff == 0 || header.e_shentsize != sizeof(Elf64_Shdr))
        {
            error = "has no section table";
            return "";
        }

        // Section 0 holds the real count and string-table index when they do not fit the header (ELF's extended
        // numbering: a library of a whole game's functions can have more than 0xFF00 sections).
        Elf64_Shdr first{};
        if (!readAt(in, fileSize, header.e_shoff, &first, sizeof first))
        {
            error = "has a section table past its end";
            return "";
        }
        const uint64_t count = header.e_shnum != 0 ? header.e_shnum : first.sh_size;
        const uint64_t names = header.e_shstrndx != SHN_XINDEX ? header.e_shstrndx : first.sh_link;
        constexpr uint64_t kMaxSections = 1u << 22;
        if (count == 0 || count > kMaxSections || names >= count)
        {
            error = "has a malformed section table";
            return "";
        }
        // Against the file's size before allocating anything for it.
        if (header.e_shoff > fileSize || count * sizeof(Elf64_Shdr) > fileSize - header.e_shoff)
        {
            error = "has a section table past its end";
            return "";
        }
        std::vector<Elf64_Shdr> sections(static_cast<size_t>(count));
        if (!readAt(in, fileSize, header.e_shoff, sections.data(), count * sizeof(Elf64_Shdr)))
        {
            error = "has a section table past its end";
            return "";
        }
        const Elf64_Shdr &nameTable = sections[static_cast<size_t>(names)];
        constexpr uint64_t kMaxNames = 64u << 20;
        if (nameTable.sh_size == 0 || nameTable.sh_size > kMaxNames || nameTable.sh_offset > fileSize ||
            nameTable.sh_size > fileSize - nameTable.sh_offset)
        {
            error = "has a malformed section name table";
            return "";
        }
        std::string nameBytes(static_cast<size_t>(nameTable.sh_size), '\0');
        if (!readAt(in, fileSize, nameTable.sh_offset, nameBytes.data(), nameTable.sh_size))
        {
            error = "has a section name table past its end";
            return "";
        }

        const size_t wantedLength = std::strlen(kSectionName);
        const Elf64_Shdr *found = nullptr;
        for (const Elf64_Shdr &section : sections)
        {
            if (section.sh_name >= nameBytes.size())
                continue;
            const char *name = nameBytes.data() + section.sh_name;
            const size_t room = nameBytes.size() - section.sh_name;
            if (room <= wantedLength || std::memcmp(name, kSectionName, wantedLength + 1) != 0)
                continue;
            if (found)
            {
                error = "has two .ghr_interface sections";
                return "";
            }
            found = &section;
        }
        if (!found)
        {
            error = "has no .ghr_interface section (not a GHR game library)";
            return "";
        }
        const size_t prefixLength = std::strlen(kSectionPrefix);
        const uint64_t wantedSize = prefixLength + kIdLength + 1;
        if (found->sh_type != SHT_PROGBITS || found->sh_size != wantedSize)
        {
            error = "has a malformed .ghr_interface section";
            return "";
        }
        std::string bytes(static_cast<size_t>(wantedSize), '\0');
        if (!readAt(in, fileSize, found->sh_offset, bytes.data(), wantedSize))
        {
            error = "has a .ghr_interface section past its end";
            return "";
        }
        if (bytes.compare(0, prefixLength, kSectionPrefix) != 0 || bytes.back() != '\0' ||
            !lowerHex(bytes.data() + prefixLength, kIdLength))
        {
            error = "has a malformed .ghr_interface section";
            return "";
        }
        error.clear();
        return bytes.substr(prefixLength, kIdLength);
    }

    Result load(const std::vector<fs::path> &paths, const std::string &expectedId)
    {
        fs::path path;
        for (const fs::path &candidate : paths)
        {
            std::error_code ec;
            if (fs::is_regular_file(candidate, ec))
            {
                path = candidate;
                break;
            }
        }
        if (path.empty())
            return refuse(Status::Missing, {}, "the game's code is not built yet (no " + std::string(kFileName) +
                                                   " in " + list(paths) + ")");

        std::string error;
        const std::string id = readInterfaceId(path, error);
        if (id.empty())
            return refuse(Status::NotAGameLibrary, path, path.string() + " " + error);
        if (id != expectedId)
            return refuse(Status::Mismatch, path,
                          path.string() + " was built for another version of GHR (its interface " + shortId(id) +
                              ", this GHR's " + shortId(expectedId) + "): build the game's code again");

        // RTLD_NOW: every symbol the library needs from GHR is resolved here, so a missing one is this error and not
        // a crash in the middle of the game. RTLD_LOCAL: its own symbols stay out of the global scope.
        dlerror();
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle)
        {
            const char *why = dlerror();
            return refuse(Status::LoadFailed, path, path.string() + " cannot be loaded: " + (why ? why : "dlopen failed"));
        }
        using EntryFunction = const Entry *(*)();
        dlerror();
        auto entryFunction = reinterpret_cast<EntryFunction>(dlsym(handle, kEntrySymbol));
        if (!entryFunction)
        {
            dlclose(handle);
            return refuse(Status::NotAGameLibrary, path, path.string() + " has no " + kEntrySymbol + "()");
        }
        const Entry *entry = entryFunction();
        if (!entry || !entry->interfaceId || std::strlen(entry->interfaceId) != kIdLength ||
            expectedId != entry->interfaceId)
        {
            const std::string said = entry && entry->interfaceId ? shortId(entry->interfaceId) : "none";
            dlclose(handle);
            return refuse(Status::Mismatch, path,
                          path.string() + "'s entry names interface " + said + ", not its section's " + shortId(id) +
                              ": build the game's code again");
        }
        if (entry->size < sizeof(Entry))
        {
            dlclose(handle);
            return refuse(Status::Mismatch, path, path.string() + " has an older entry layout: build the game's code again");
        }
        if (!entry->functions || entry->slotCount == 0 || entry->end < entry->base ||
            entry->end - entry->base != entry->slotCount * 4ull || !entry->modulePacks)
        {
            dlclose(handle);
            return refuse(Status::NotAGameLibrary, path, path.string() + " has an inconsistent function table");
        }
        Result r;
        r.status = Status::Loaded;
        r.path = path;
        r.entry = entry;
        r.handle = handle;
        return r; // the handle stays open: the game runs from it until the process ends
    }

#if KAMITE_SPLIT_GAME_CODE
    namespace
    {
        const Entry *g_used = nullptr;
    }

    void use(const Entry *entry)
    {
        g_used = entry;
    }

    bool registerVu1Programs()
    {
        if (!g_used || !g_used->registerVu1Programs)
            return false;
        g_used->registerVu1Programs();
        return true;
    }
#endif
}

#if KAMITE_SPLIT_GAME_CODE
namespace kamite
{
    // module_registry.h's list, which the normal build generates into GHR (kamite_generated/kamite/module_index.cpp)
    // and the split build into the library: this one hands out the loaded library's. No library, no modules.
    const ModulePack *const *modulePacks(size_t &count)
    {
        if (game_library::g_used && game_library::g_used->modulePacks)
            return game_library::g_used->modulePacks(count);
        static const ModulePack *const none[] = {nullptr};
        count = 0;
        return none;
    }
}
#endif
