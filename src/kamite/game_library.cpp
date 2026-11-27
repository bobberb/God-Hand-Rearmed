// The split game code's loader (game_library.h).
//
// Reading the interface id from the file needs no operating system: an ELF reader and a PE reader on the formats'
// own layouts (declared here, so both compile and are tested on every platform). Loading needs one: dlopen on Linux,
// LoadLibraryExW on Windows (osOpen / osSymbol / osClose below).

#include "kamite/game_library.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstring>
#include <cwchar>
#include <iterator>
#include <fstream>
#include <limits>
#include <system_error>

namespace fs = std::filesystem;

namespace kamite::game_library
{
    std::string pathText(const fs::path &path)
    {
        const std::u8string text = path.u8string();
        return std::string(text.begin(), text.end());
    }

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
                out += (out.empty() ? "" : ", ") + pathText(p);
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

        // The id section's bytes: the prefix, 64 lowercase hex digits, NUL. The id, or "" and why in `error`.
        std::string idFromSection(const std::string &bytes, const char *sectionName, std::string &error)
        {
            const size_t prefixLength = std::strlen(kSectionPrefix);
            if (bytes.size() < prefixLength + kIdLength + 1 || bytes.compare(0, prefixLength, kSectionPrefix) != 0 ||
                bytes[prefixLength + kIdLength] != '\0' || !lowerHex(bytes.data() + prefixLength, kIdLength))
            {
                error = std::string("has a malformed ") + sectionName + " section";
                return "";
            }
            error.clear();
            return bytes.substr(prefixLength, kIdLength);
        }

        // ELF64, little-endian: the fields read here, as the format lays them out (elf.h's Elf64_Ehdr / Elf64_Shdr).
#pragma pack(push, 1)
        struct ElfHeader
        {
            unsigned char ident[16];
            uint16_t type, machine;
            uint32_t version;
            uint64_t entry, phoff, shoff;
            uint32_t flags;
            uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
        };
        struct ElfSection
        {
            uint32_t name, type;
            uint64_t flags, addr, offset, size;
            uint32_t link, info;
            uint64_t addralign, entsize;
        };
#pragma pack(pop)
        static_assert(sizeof(ElfHeader) == 64 && sizeof(ElfSection) == 64, "ELF64 layouts");
        constexpr uint16_t kElfShared = 3;            // ET_DYN
        constexpr uint32_t kElfProgbits = 1;          // SHT_PROGBITS
        constexpr uint16_t kElfExtendedIndex = 0xFFFF; // SHN_XINDEX

        std::string readElf(std::ifstream &in, uint64_t fileSize, std::string &error)
        {
            ElfHeader header{};
            if (!readAt(in, fileSize, 0, &header, sizeof header))
            {
                error = "is not an ELF file";
                return "";
            }
            if (header.ident[4] != 2 || header.ident[5] != 1) // ELFCLASS64, ELFDATA2LSB
            {
                error = "is not a 64-bit little-endian ELF file";
                return "";
            }
            if (header.type != kElfShared)
            {
                error = "is not a shared library";
                return "";
            }
            if (header.shoff == 0 || header.shentsize != sizeof(ElfSection))
            {
                error = "has no section table";
                return "";
            }
            // Section 0 holds the real count and string-table index when they do not fit the header (ELF's extended
            // numbering: a library of a whole game's functions can have more than 0xFF00 sections).
            ElfSection first{};
            if (!readAt(in, fileSize, header.shoff, &first, sizeof first))
            {
                error = "has a section table past its end";
                return "";
            }
            const uint64_t count = header.shnum != 0 ? header.shnum : first.size;
            const uint64_t names = header.shstrndx != kElfExtendedIndex ? header.shstrndx : first.link;
            constexpr uint64_t kMaxSections = 1u << 22;
            if (count == 0 || count > kMaxSections || names >= count)
            {
                error = "has a malformed section table";
                return "";
            }
            // Against the file's size before allocating anything for it.
            if (header.shoff > fileSize || count * sizeof(ElfSection) > fileSize - header.shoff)
            {
                error = "has a section table past its end";
                return "";
            }
            std::vector<ElfSection> sections(static_cast<size_t>(count));
            if (!readAt(in, fileSize, header.shoff, sections.data(), count * sizeof(ElfSection)))
            {
                error = "has a section table past its end";
                return "";
            }
            const ElfSection &nameTable = sections[static_cast<size_t>(names)];
            constexpr uint64_t kMaxNames = 64u << 20;
            if (nameTable.size == 0 || nameTable.size > kMaxNames || nameTable.offset > fileSize ||
                nameTable.size > fileSize - nameTable.offset)
            {
                error = "has a malformed section name table";
                return "";
            }
            std::string nameBytes(static_cast<size_t>(nameTable.size), '\0');
            if (!readAt(in, fileSize, nameTable.offset, nameBytes.data(), nameTable.size))
            {
                error = "has a section name table past its end";
                return "";
            }
            const size_t wantedLength = std::strlen(kElfSectionName);
            const ElfSection *found = nullptr;
            for (const ElfSection &section : sections)
            {
                if (section.name >= nameBytes.size())
                    continue;
                const size_t room = nameBytes.size() - section.name;
                if (room <= wantedLength || std::memcmp(nameBytes.data() + section.name, kElfSectionName, wantedLength + 1) != 0)
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
            const uint64_t wantedSize = std::strlen(kSectionPrefix) + kIdLength + 1;
            if (found->type != kElfProgbits || found->size != wantedSize)
            {
                error = "has a malformed .ghr_interface section";
                return "";
            }
            std::string bytes(static_cast<size_t>(wantedSize), '\0');
            if (!readAt(in, fileSize, found->offset, bytes.data(), wantedSize))
            {
                error = "has a .ghr_interface section past its end";
                return "";
            }
            return idFromSection(bytes, ".ghr_interface", error);
        }

        // PE32+ (x64): the fields read here (winnt.h's IMAGE_FILE_HEADER / IMAGE_SECTION_HEADER).
#pragma pack(push, 1)
        struct PeFileHeader
        {
            uint16_t machine, numberOfSections;
            uint32_t timeDateStamp, pointerToSymbolTable, numberOfSymbols;
            uint16_t sizeOfOptionalHeader, characteristics;
        };
        struct PeSection
        {
            char name[8];
            uint32_t virtualSize, virtualAddress, sizeOfRawData, pointerToRawData;
            uint32_t pointerToRelocations, pointerToLinenumbers;
            uint16_t numberOfRelocations, numberOfLinenumbers;
            uint32_t characteristics;
        };
#pragma pack(pop)
        static_assert(sizeof(PeFileHeader) == 20 && sizeof(PeSection) == 40, "PE layouts");
        constexpr uint16_t kPeAmd64 = 0x8664;  // IMAGE_FILE_MACHINE_AMD64
        constexpr uint16_t kPeDll = 0x2000;    // IMAGE_FILE_DLL
        constexpr uint16_t kPe32Plus = 0x20B;  // IMAGE_NT_OPTIONAL_HDR64_MAGIC

        std::string readPe(std::ifstream &in, uint64_t fileSize, std::string &error)
        {
            uint32_t peOffset = 0; // e_lfanew, at 0x3C of the DOS header
            char signature[4] = {};
            PeFileHeader header{};
            if (!readAt(in, fileSize, 0x3C, &peOffset, sizeof peOffset) || !readAt(in, fileSize, peOffset, signature, 4) ||
                std::memcmp(signature, "PE\0\0", 4) != 0 || !readAt(in, fileSize, uint64_t(peOffset) + 4, &header, sizeof header))
            {
                error = "is not a PE file";
                return "";
            }
            uint16_t optionalMagic = 0;
            const uint64_t optionalOffset = uint64_t(peOffset) + 4 + sizeof header;
            if (header.machine != kPeAmd64 || !(header.characteristics & kPeDll) ||
                !readAt(in, fileSize, optionalOffset, &optionalMagic, sizeof optionalMagic) || optionalMagic != kPe32Plus)
            {
                error = "is not a 64-bit Windows DLL";
                return "";
            }
            const uint64_t tableOffset = optionalOffset + header.sizeOfOptionalHeader;
            const uint64_t count = header.numberOfSections;
            if (count == 0 || tableOffset > fileSize || count * sizeof(PeSection) > fileSize - tableOffset)
            {
                error = "has a section table past its end";
                return "";
            }
            std::vector<PeSection> sections(static_cast<size_t>(count));
            if (!readAt(in, fileSize, tableOffset, sections.data(), count * sizeof(PeSection)))
            {
                error = "has a section table past its end";
                return "";
            }
            // The name is 8 bytes, NUL-padded (".ghrid" fits; a longer one would be truncated in an image).
            char wanted[8] = {};
            std::memcpy(wanted, kPeSectionName, std::strlen(kPeSectionName));
            const PeSection *found = nullptr;
            for (const PeSection &section : sections)
            {
                if (std::memcmp(section.name, wanted, sizeof wanted) != 0)
                    continue;
                if (found)
                {
                    error = "has two .ghrid sections";
                    return "";
                }
                found = &section;
            }
            if (!found)
            {
                error = "has no .ghrid section (not a GHR game library)";
                return "";
            }
            // VirtualSize is the data's own size; SizeOfRawData is it rounded up to the file alignment.
            const uint64_t wantedSize = std::strlen(kSectionPrefix) + kIdLength + 1;
            if (found->virtualSize < wantedSize || found->sizeOfRawData < wantedSize)
            {
                error = "has a malformed .ghrid section";
                return "";
            }
            std::string bytes(static_cast<size_t>(wantedSize), '\0');
            if (!readAt(in, fileSize, found->pointerToRawData, bytes.data(), wantedSize))
            {
                error = "has a .ghrid section past its end";
                return "";
            }
            return idFromSection(bytes, ".ghrid", error);
        }

        // The operating system's half: open, look up, close. The error text for the log when open fails.
#if defined(_WIN32)
        std::string lastError()
        {
            const DWORD code = GetLastError();
            char *text = nullptr;
            FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                           nullptr, code, 0, reinterpret_cast<char *>(&text), 0, nullptr);
            std::string message = text ? text : "LoadLibrary failed";
            if (text)
                LocalFree(text);
            while (!message.empty() && (message.back() == '\n' || message.back() == '\r' || message.back() == ' '))
                message.pop_back();
            return message + " (error " + std::to_string(code) + ")";
        }

        void *osOpen(const fs::path &path, std::string &error)
        {
            // The library imports the runtime from a module named GHR.exe (GHR.lib, src/CMakeLists.txt). In a program
            // under another name that import would not bind to this process: Windows would look for a GHR.exe to load
            // (another copy of the runtime, never initialized) or fail. Refused with the reason instead.
            wchar_t self[32768] = {};
            const DWORD length = GetModuleFileNameW(nullptr, self, static_cast<DWORD>(std::size(self)));
            const std::wstring name = fs::path(std::wstring(self, length)).filename().wstring();
            if (length == 0 || _wcsicmp(name.c_str(), L"GHR.exe") != 0)
            {
                error = "the game library imports from GHR.exe, and this program is " + pathText(fs::path(name)) +
                        ": keep the program's name GHR.exe";
                return nullptr;
            }
            // Every import (from GHR.exe and the C runtime) is resolved here, as RTLD_NOW does; the library's own
            // dependencies are looked for next to it (LOAD_WITH_ALTERED_SEARCH_PATH wants backslashes).
            fs::path native = path;
            native.make_preferred();
            HMODULE module = LoadLibraryExW(native.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (!module)
                error = lastError();
            return reinterpret_cast<void *>(module);
        }

        void *osSymbol(void *handle, const char *name)
        {
            return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
        }

        void osClose(void *handle)
        {
            FreeLibrary(reinterpret_cast<HMODULE>(handle));
        }
#else
        void *osOpen(const fs::path &path, std::string &error)
        {
            // RTLD_NOW: every symbol the library needs from GHR is resolved here, so a missing one is this error and
            // not a crash in the middle of the game. RTLD_LOCAL: its own symbols stay out of the global scope.
            dlerror();
            void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle)
            {
                const char *why = dlerror();
                error = why ? why : "dlopen failed";
            }
            return handle;
        }

        void *osSymbol(void *handle, const char *name)
        {
            dlerror();
            return dlsym(handle, name);
        }

        void osClose(void *handle)
        {
            dlclose(handle);
        }
#endif
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
            // Absolute: dlopen and LoadLibrary search their paths for a bare name, so a relative name could be checked
            // here as one file and loaded as another.
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
        char magic[4] = {};
        if (readAt(in, fileSize, 0, magic, 4) && std::memcmp(magic, "\x7f" "ELF", 4) == 0)
            return readElf(in, fileSize, error);
        if (readAt(in, fileSize, 0, magic, 2) && magic[0] == 'M' && magic[1] == 'Z')
            return readPe(in, fileSize, error);
        error = "is not a shared library (neither an ELF nor a PE file)";
        return "";
    }

    void *findSymbol(void *handle, const char *name)
    {
        return handle && name ? osSymbol(handle, name) : nullptr;
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
            return refuse(Status::NotAGameLibrary, path, pathText(path) + " " + error);
        if (id != expectedId)
            return refuse(Status::Mismatch, path,
                          pathText(path) + " was built for another version of GHR (its interface " + shortId(id) +
                              ", this GHR's " + shortId(expectedId) + "): build the game's code again");

        void *handle = osOpen(path, error);
        if (!handle)
            return refuse(Status::LoadFailed, path, pathText(path) + " cannot be loaded: " + error);
        using EntryFunction = const Entry *(*)();
        auto entryFunction = reinterpret_cast<EntryFunction>(osSymbol(handle, kEntrySymbol));
        if (!entryFunction)
        {
            osClose(handle);
            return refuse(Status::NotAGameLibrary, path, pathText(path) + " has no " + kEntrySymbol + "()");
        }
        const Entry *entry = entryFunction();
        if (!entry || !entry->interfaceId || std::strlen(entry->interfaceId) != kIdLength ||
            expectedId != entry->interfaceId)
        {
            const std::string said = entry && entry->interfaceId ? shortId(entry->interfaceId) : "none";
            osClose(handle);
            return refuse(Status::Mismatch, path,
                          pathText(path) + "'s entry names interface " + said + ", not its section's " + shortId(id) +
                              ": build the game's code again");
        }
        if (entry->size < sizeof(Entry))
        {
            osClose(handle);
            return refuse(Status::Mismatch, path, pathText(path) + " has an older entry layout: build the game's code again");
        }
        if (!entry->functions || entry->slotCount == 0 || entry->end < entry->base ||
            entry->end - entry->base != entry->slotCount * 4ull || !entry->modulePacks)
        {
            osClose(handle);
            return refuse(Status::NotAGameLibrary, path, pathText(path) + " has an inconsistent function table");
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
