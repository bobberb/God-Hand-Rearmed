"""Gate: state the game code shares with the runtime lives in one place, on every platform.

    python -I tools/linux/check_shared_state.py [--list] [<repo root>]

With the split game code (src/kamite/game_library.h) the generated code is a library and the runtime is in GHR. On
Linux the dynamic linker makes a header-defined static one object in both; on Windows it never does: a DLL keeps its own
copy of every inline variable, every static local of an inline function, every static data member defined in a header.
State the runtime writes and the game code reads (the scratchpad's host pointer, the VU1 packet sink) would then be
null or stale in the game code.

So the headers the game code compiles against must hold no such definition, unless it is
  - inside `#if !PS2X_SHARED_STATE` (the normal build's form, with the shared form in the #else), or
  - marked on its line with `// shared-state: <why one copy per module is fine>` (an environment flag read once, a
    constant table, a debug counter).
Everything else fails the gate, with file:line.

The headers: every header and .inl under the runtime's and the IOP's include folders and the runtime's vu/ and
Kernel/Stubs/ folders (the interface id's set, src/CMakeLists.txt), plus src/kamite/module_registry.h.
Exit 0 = clean, 1 = findings (printed), 2 = usage.
"""
import re
import sys
from pathlib import Path

FOLDERS = [
    "runtime/ps2xRuntime/include",
    "runtime/ps2xIOP/include",
    "runtime/ps2xRuntime/src/lib/vu",
    "runtime/ps2xRuntime/src/lib/Kernel/Stubs",
]
FILES = ["src/kamite/module_registry.h"]
SUFFIXES = {".h", ".hpp", ".inl"}
MARKER = "// shared-state:"

# A definition with storage that each module would get its own copy of.
INLINE_VARIABLE = re.compile(r"^\s*(?:static\s+)?inline\s+(?:thread_local\s+)?(?!constexpr\b)(?!namespace\b)"
                             r"[A-Za-z_][\w:<>,\s\*&]*?\b[A-Za-z_]\w*\s*(?:\[[^\]]*\])?\s*(?:=|\{|;)")
STATIC_LOCAL = re.compile(r"^\s+static\s+(?:thread_local\s+)?(?!constexpr\b)(?!inline\b)(?!void\b)(?!assert)"
                          r"[A-Za-z_][\w:<>,\s\*&]*?\b[A-Za-z_]\w*\s*(?:\[[^\]]*\])*\s*(?:=|\{|;)")


def strip_comments(line):
    return re.sub(r"//.*", "", line)


def scan(path: Path, root: Path):
    """[(line number, text, kind)] of unmarked per-module definitions outside the normal-build-only branch."""
    findings = []
    depth = 0          # brace depth (static locals only count inside a function body: depth > 0 and in a function)
    condition = []     # stack of booleans: inside `#if !PS2X_SHARED_STATE`'s first branch
    in_block_comment = False
    function_depths = []  # brace depths at which a function body opened
    pending_function = False
    for number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw
        if in_block_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_block_comment = False
            else:
                continue
        while "/*" in line and "*/" not in line.split("/*", 1)[1]:
            line = line.split("/*", 1)[0]
            in_block_comment = True
        line = re.sub(r"/\*.*?\*/", "", line)
        stripped = line.strip()
        if stripped.startswith("#"):
            directive = stripped[1:].strip()
            if directive.startswith(("if", "ifdef", "ifndef")):
                condition.append(bool(re.match(r"if\s+!\s*PS2X_SHARED_STATE\b", directive)))
            elif directive.startswith(("else", "elif")) and condition:
                condition[-1] = False
            elif directive.startswith("endif") and condition:
                condition.pop()
            continue
        code = strip_comments(line)
        normal_only = any(condition)
        marked = MARKER in raw
        if not normal_only and not marked and re.match(r'^\s*extern\s+(?!"C")', code) \
                and "(" not in code.split("[")[0] and "PS2X_RUNTIME_API" not in code:
            # A variable defined once in the runtime and read by the game code: Windows imports data only through
            # __declspec(dllimport), which PS2X_RUNTIME_API gives it.
            findings.append((number, raw.strip(), "extern variable without PS2X_RUNTIME_API"))
        elif not normal_only and not marked:
            inside_function = bool(function_depths) and depth >= function_depths[-1]
            if depth == 0 or not inside_function:
                if INLINE_VARIABLE.match(code) and "(" not in code.split("=")[0]:
                    findings.append((number, raw.strip(), "inline variable"))
                elif re.match(r"^\s*static\s+(?:inline\s+)?(?!constexpr)[\w:<>,\s\*&]+\b\w+\s*(?:=|\{|;)", code) \
                        and "(" not in code.split("=")[0] and depth > 0:
                    findings.append((number, raw.strip(), "static data member"))
            elif STATIC_LOCAL.match(code) and "(" not in code.split("=")[0].split("{")[0]:
                findings.append((number, raw.strip(), "static local of a header function"))
        # Track function bodies: a line with ')' then '{' (or a '{' line after such a header) opens one.
        if re.search(r"\)\s*(const\s*)?(noexcept\s*)?(->\s*[\w:<>\*&\s]+)?\s*\{", code) or \
                (pending_function and stripped.startswith("{")):
            function_depths.append(depth + 1)
            pending_function = False
        elif re.search(r"\)\s*(const\s*)?(noexcept\s*)?$", code) and not code.rstrip().endswith(";"):
            pending_function = True
        elif stripped and not stripped.startswith("{"):
            pending_function = False
        for ch in code:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                while function_depths and depth < function_depths[-1]:
                    function_depths.pop()
    return findings


def headers(root: Path):
    for folder in FOLDERS:
        for path in sorted((root / folder).rglob("*")):
            if path.suffix in SUFFIXES and path.is_file():
                yield path
    for name in FILES:
        if (root / name).is_file():
            yield root / name


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) > 1:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    root = Path(args[0] if args else Path(__file__).resolve().parents[2])
    total = 0
    for path in headers(root):
        for number, text, kind in scan(path, root):
            total += 1
            print(f"{path.relative_to(root)}:{number}: {kind}: {text[:110]}")
    if total:
        print(f"{total} per-module definition(s) in the game code's headers: move each behind PS2X_SHARED_STATE or mark "
              f"it '{MARKER} <why>' (tools/linux/check_shared_state.py)")
        return 1
    print("shared state: no per-module definitions in the game code's headers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
