"""Gate, compiler half: the shared data the game code's headers define, as the compiler emits it.

    python -I tools/linux/check_shared_symbols.py <nm> <object> <allow-list>

<object> is tests/unit/shared_state_tu.cpp compiled with the split build's definitions and -fkeep-inline-functions
(src/CMakeLists.txt, target ghr_shared_state_tu): every header the game code includes, every inline function emitted.
A data symbol it defines with vague linkage (nm types u, V, v: an inline variable, a static local of an inline
function, a template's static) or as a strong global (B, D, G, S) is one copy per module on Windows. Each must match
an entry of <allow-list> (tools/linux/shared_state_allowlist.txt: a regular expression on the demangled name, then
why one copy per module is fine). Anything else fails, as do allow-list entries nothing matches any more (stale).

The source half, tools/linux/check_shared_state.py, reads the headers' lines and catches what the compiler does not
emit here (an inline variable nothing in the headers uses). Exit 0 = clean, 1 = findings, 2 = usage.
"""
import re
import subprocess
import sys

SHARED_TYPES = {"u", "V", "v", "B", "D", "G", "S"}


def shared_symbols(nm, obj):
    out = subprocess.run([nm, "-C", "--defined-only", obj], check=True, capture_output=True, text=True).stdout
    symbols = []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[1] in SHARED_TYPES:
            symbols.append(parts[2])
    return sorted(set(symbols))


def read_allow_list(path):
    entries = []
    for number, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        pattern, _, why = line.partition("  # ")
        if not why.strip():
            raise SystemExit(f"{path}:{number}: an entry needs '  # <why one copy per module is fine>'")
        entries.append((number, re.compile(pattern.strip()), why.strip()))
    return entries


def main(argv):
    if len(argv) != 4:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    nm, obj, allow_path = argv[1:]
    symbols = shared_symbols(nm, obj)
    functions = subprocess.run([nm, "--defined-only", obj], check=True, capture_output=True, text=True).stdout.count(" W ")
    if functions < 100:
        print(f"only {functions} inline functions emitted: was the object built with -fkeep-inline-functions?")
        return 1
    entries = read_allow_list(allow_path)
    used = set()
    failures = []
    for symbol in symbols:
        match = next((n for n, pattern, _ in entries if pattern.fullmatch(symbol)), None)
        if match is None:
            failures.append(symbol)
        else:
            used.add(match)
    for symbol in failures:
        print(f"shared data, one copy per module on Windows: {symbol}")
    stale = [(n, p.pattern) for n, p, _ in entries if n not in used]
    for n, pattern in stale:
        print(f"{allow_path}:{n}: matches nothing any more: {pattern}")
    print(f"{len(symbols)} shared data symbols from {functions} inline functions: {len(symbols) - len(failures)} allowed, "
          f"{len(failures)} not; {len(stale)} stale allow-list entries")
    return 1 if failures or stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
