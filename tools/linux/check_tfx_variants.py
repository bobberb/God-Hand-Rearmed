#!/usr/bin/env python3
"""Compiles the Vulkan renderer's main shader (tfx.glsl) in the variants the game builds, so a broken edit fails CI
instead of a player's GPU. The `tfx-variants` Nix check (flake.nix) runs it with tfxcc (tools/linux/tfxcc: the
game's own libshaderc, called with the game's options).

    python3 -I tools/linux/check_tfx_variants.py --compiler tfxcc [--tfx scratch/tfx.glsl] [--list] [--dump DIR]

Everything is read from the sources, nothing is listed by hand:
  - how a variant's source is put together: GSDeviceVK.cpp, AddShaderHeader (its lines and the device features that
    guard them), AddShaderStageMacro, and GetTFX{Vertex,Fragment}Shader (the AddMacro calls, in order, then the shader
    text). A statement there this script does not understand stops it with an error rather than being skipped.
  - each macro's possible values: the selector field it is set from (GSDevice.h, VSSelector / PSSelector), every
    value of its bit width, or of its enum (GSShaderEnums.h); a device feature (`features.x`, `m_features.x`,
    `dev->X()`) is 0 or 1.
  - which of those values differ: every place tfx.glsl compares the macro with a constant (==, !=, <, <=, >, >=, &,
    or a bare test, which is != 0). Values giving the same answer to all of them are one class and the smallest is
    compiled. A macro compared with another macro keeps all its values; one used as a number in code also keeps its
    largest.

Coverage rule, per stage (vert, frag), over the parameters (selector macros and device features): every value class
of every parameter, every pair of classes of every two parameters, and every triple of classes of every three
parameters that one block of tfx.glsl is conditioned on (its #if / #elif and all those around it, with the shader's
own #defines like SW_BLEND or ZWRITE and the header's like DISABLE_TEXTURE_BARRIER expanded to what they read)
appears in at least one variant: a greedy covering array, deterministic. Combinations no game build can contain are left out by CONSTRAINTS below, each citing the game code
that rules it out; one the greedy fill could not place in any variant is an error. Variants whose assembled source
is the same text are compiled once.

Exit 0 when every variant compiles, 1 naming each failing variant (stage, non-zero macros, features) with the
compiler's messages, their line numbers also given as tfx.glsl lines; 2 for a problem reading the sources.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import itertools
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
DEVICE_CPP = "thirdparty/pcsx2/pcsx2/GS/Renderers/Vulkan/GSDeviceVK.cpp"
DEVICE_H = "thirdparty/pcsx2/pcsx2/GS/Renderers/Common/GSDevice.h"
ENUMS_H = "thirdparty/pcsx2/pcsx2/GS/Renderers/Common/GSShaderEnums.h"
RENDERER_HW = "thirdparty/pcsx2/pcsx2/GS/Renderers/HW/GSRendererHW.cpp"
TFX = "thirdparty/pcsx2-gs/resources/shaders/vulkan/tfx.glsl"
BRIDGE = "thirdparty/pcsx2-gs/compat/Bridge.cpp"


class SourceError(Exception):
    """The C++ or the shader has a shape this script does not know; fix the script, never skip."""


# ---------------------------------------------------------------------------------------------------------------------
# Constraints: combinations the game never builds. `anchor` is a regex that must match exactly one line of `path`
# (test_check_tfx_variants.py checks it), so the file:line printed stays right when the file moves.


@dataclasses.dataclass(frozen=True)
class Constraint:
    stage: str
    params: tuple[str, ...]
    ok: Callable[[dict], bool]
    why: str
    path: str
    anchor: str


CONSTRAINTS = [
    Constraint("frag", ("PS_ROV_COLOR", "PS_ROV_DEPTH", "features.rov"),
               lambda v: not (v["PS_ROV_COLOR"] or v["PS_ROV_DEPTH"]) or v["features.rov"],
               "ROVs only on a device with ROV support (the header's interlock extensions come with it)",
               RENDERER_HW, r"if \(!\(GSConfig\.HWROV && features\.rov\)\)"),
    Constraint("frag", ("PS_ROV_COLOR", "PS_NO_COLOR1"),
               lambda v: not v["PS_ROV_COLOR"] or v["PS_NO_COLOR1"],
               "colour ROV blends in the shader and drops the dual-source output",
               RENDERER_HW, r"// We should never need dual source with SW blend"),
    Constraint("frag", ("PS_NO_COLOR", "PS_NO_COLOR1"),
               lambda v: not v["PS_NO_COLOR"] or v["PS_NO_COLOR1"],
               "no colour output means no second colour output either",
               DEVICE_H, r"no_color = no_color1 = 1;"),
    Constraint("frag", ("PS_DATE",),
               lambda v: v["PS_DATE"] in (0, 1, 2, 3, 5, 6),
               "DATE is 1 + DATM (PrimID), 5 + DATM (barrier) or 3 (after the PrimID pass); DATM is one bit",
               RENDERER_HW, r"m_conf\.ps\.date = 1 \+ m_cached_ctx\.TEST\.DATM;"),
    Constraint("frag", ("PS_ANISOTROPIC_FILTERING",),
               lambda v: v["PS_ANISOTROPIC_FILTERING"] <= 16,
               "sw_aniso is GSConfig.MaxAnisotropy or 0, and the launcher's option is clamped to 0..16",
               BRIDGE, r"options\.MaxAnisotropy = static_cast<u8>\(std::clamp\(value, 0, 16\)\);"),
    Constraint("frag", ("PS_DATE", "PS_NO_COLOR"),
               lambda v: v["PS_DATE"] not in (1, 2) or not v["PS_NO_COLOR"],
               "DATE 1 and 2 are the PrimID pass, which writes colour",
               DEVICE_CPP, r"pipe\.ps\.no_color = false;"),
]


def cite(c: Constraint, root: Path = ROOT) -> str:
    lines = (root / c.path).read_text().splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if re.search(c.anchor, line)]
    if len(hits) != 1:
        raise SourceError(f"constraint anchor {c.anchor!r} matches {len(hits)} lines of {c.path}, want 1")
    return f"{c.path}:{hits[0]}"


# ---------------------------------------------------------------------------------------------------------------------
# Reading the C++.


def strip_c_comments(text: str) -> str:
    """Comments to spaces, keeping line numbers (newlines inside /* */ stay)."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', lambda m: m.group(0) if m.group(0)[0] == '"' else blank(m),
                  text, flags=re.S)


def braced(text: str, start: int) -> str:
    """What is between the brace at `start` and the one closing it."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1:i]
    raise SourceError("unbalanced braces")


def function_body(text: str, signature: str) -> tuple[str, int]:
    """The body (between the outer braces) of the one function whose definition matches `signature`, and the line
    number of its opening brace."""
    hits = [m for m in re.finditer(signature, text) if text[m.end():].lstrip().startswith("{")]
    if len(hits) != 1:
        raise SourceError(f"{len(hits)} definitions match {signature!r}, want 1")
    start = text.index("{", hits[0].end())
    return braced(text, start), text.count("\n", 0, start) + 1


def c_string(literal: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}[m.group(1)], literal)


def feature_expr(cond: str, where: str) -> tuple[str, set[str]]:
    """A C++ guard over device features (`features.x`, `m_features.x`, `dev->X()`) as a Python expression over F, and
    the feature names in it."""
    names = set()

    def feature(m):
        names.add(f"features.{m.group(1) or m.group(2)}")
        return f' F["features.{m.group(1) or m.group(2)}"] '
    py = re.sub(r"\b(?:m_)?features\.(\w+)|\bdev->(\w+)\(\)", feature, cond)
    py = re.sub(r"!(?!=)", " not ", py.replace("&&", " and ").replace("||", " or "))
    if re.sub(r'F\["[\w.]+"\]|\b(?:and|or|not)\b|[()\s]', "", py):
        raise SourceError(f"{where}: guard {cond!r} uses something other than device features")
    return py, names


@dataclasses.dataclass
class HeaderLine:
    text: str
    guard: str | None  # Python expression over F, None = always


def parse_header(cpp: str) -> tuple[list[HeaderLine], set[str]]:
    """AddShaderHeader: `ss << "...";` lines, each under the `if (...)` guarding it (one statement or a braced block)."""
    body, line0 = function_body(cpp, r"static void AddShaderHeader\(std::stringstream& ss\)")
    lines, features, guard_stack, pending = [], set(), [], None
    for n, raw in enumerate(body.splitlines(), line0):
        s = raw.strip()
        where = f"{DEVICE_CPP}:{n}"
        if not s:
            continue
        m = re.fullmatch(r"if \((.+)\)", s)
        if m:
            pending, used = feature_expr(m.group(1), where)
            features |= used
            continue
        if s == "{":
            guard_stack.append(pending)
            pending = None
            continue
        if s == "}":
            guard_stack.pop()
            continue
        m = re.fullmatch(r'ss << "((?:\\.|[^"\\])*)";', s)
        if m:
            guards = [g for g in guard_stack + [pending] if g]
            lines.append(HeaderLine(c_string(m.group(1)), " and ".join(f"({g})" for g in guards) or None))
            pending = None
            continue
        if re.fullmatch(r"const [\w:<>]+\*? ?\w+ = [^;]+;", s):  # the dev / features locals
            continue
        raise SourceError(f"{where}: AddShaderHeader statement {s!r} is not understood; teach {Path(__file__).name}")
    return lines, features


def parse_add_macro(cpp: str) -> list[str]:
    """AddMacro's output as pieces: string literals, and "{name}" / "{value}" where it streams its parameters. Its
    one statement must be `ss << ...;` of literals, `name` and `value`, with an integer `value`."""
    m = re.search(r"static void AddMacro\(std::stringstream& ss, const char\* name, (\w+) value\)", cpp)
    if not m or m.group(1) not in ("int", "u32", "s32", "unsigned"):
        raise SourceError(f"{DEVICE_CPP}: AddMacro(ss, const char* name, <integer> value) not found")
    body, line0 = function_body(cpp, re.escape(m.group(0)))
    statement = re.fullmatch(r"\s*ss((?:\s*<<\s*(?:\"(?:\\.|[^\"\\])*\"|name|value))+)\s*;\s*", body)
    if not statement:
        raise SourceError(f"{DEVICE_CPP}:{line0}: AddMacro is not one `ss << ...;` of literals, name and value; "
                          f"teach {Path(__file__).name}")
    pieces = []
    for piece in re.findall(r'<<\s*("(?:\\.|[^"\\])*"|name|value)', statement.group(1)):
        pieces.append("{" + piece + "}" if piece in ("name", "value") else c_string(piece[1:-1]).replace("{", "{{")
                      .replace("}", "}}"))
    if pieces.count("{name}") != 1 or pieces.count("{value}") != 1:
        raise SourceError(f"{DEVICE_CPP}:{line0}: AddMacro must stream name and value once each")
    return pieces


def parse_stage_macros(cpp: str) -> dict[str, str]:
    """AddShaderStageMacro's line for each of its flags (vs, gs, fs): an if / else if chain, one `ss << "...";` under
    each flag. Any other statement (an unconditional line, a brace, a second line under a flag) is a SourceError."""
    body, line0 = function_body(cpp, r"static void AddShaderStageMacro\(std::stringstream& ss, bool vs, bool gs, bool fs\)")
    found, flag = {}, None
    for n, raw in enumerate(body.splitlines(), line0):
        s = raw.strip()
        if not s:
            continue
        m = re.fullmatch(r"(else )?if \((vs|gs|fs)\)", s)
        if m and flag is None and bool(m.group(1)) == bool(found) and m.group(2) not in found:
            flag = m.group(2)
            continue
        m = re.fullmatch(r'ss << "((?:\\.|[^"\\])*)";', s)
        if m and flag is not None:
            found[flag], flag = c_string(m.group(1)), None
            continue
        raise SourceError(f"{DEVICE_CPP}:{n}: AddShaderStageMacro statement {s!r} is not understood; "
                          f"teach {Path(__file__).name}")
    if set(found) != {"vs", "gs", "fs"} or flag is not None:
        raise SourceError(f"AddShaderStageMacro: want lines for vs, gs, fs, found {sorted(found)}")
    return found


@dataclasses.dataclass
class Macro:
    name: str
    field: str | None  # selector field, or
    feature: str | None  # device feature ("features.x")
    line: int


@dataclasses.dataclass
class StageRecipe:
    stage: str  # "vert" / "frag"
    function: str
    selector: str  # "VSSelector" / "PSSelector"
    stage_line: str  # AddShaderStageMacro's line
    macros: list[Macro]


KINDS = {"GetVertexShader": "vert", "GetFragmentShader": "frag"}


def parse_macro_arg(expr: str, where: str) -> tuple[str | None, str | None]:
    e = re.sub(r"^static_cast<\w+>\((.*)\)$", r"\1", expr.strip())
    m = re.fullmatch(r"sel\.(\w+)", e)
    if m:
        return m.group(1), None
    m = re.fullmatch(r"m_features\.(\w+)", e)
    if m:
        return None, f"features.{m.group(1)}"
    raise SourceError(f"{where}: AddMacro value {expr!r} is neither a selector field nor a device feature; "
                      f"teach {Path(__file__).name}")


def parse_stages(cpp: str) -> list[StageRecipe]:
    """Each GSDeviceVK::GetTFX*Shader: header, stage macro, AddMacro lines in order, then the shader text."""
    stage_lines = parse_stage_macros(cpp)
    recipes = []
    for m in re.finditer(r"\w+ GSDeviceVK::(GetTFX\w+Shader)\((?:const )?GSHWDrawConfig::(\w+)&? sel\)", cpp):
        function, selector = m.group(1), m.group(2)
        body, line0 = function_body(cpp, re.escape(m.group(0)))
        steps, macros, kind = [], [], None
        for n, raw in enumerate(body.splitlines(), line0):
            s = raw.strip()
            where = f"{DEVICE_CPP}:{n}"
            if s == "AddShaderHeader(ss);":
                steps.append("header")
            elif mm := re.fullmatch(r"AddShaderStageMacro\(ss, (true|false), (true|false), (true|false)\);", s):
                flags = [k for k, f in zip(("vs", "gs", "fs"), mm.groups()) if f == "true"]
                steps.append("stage")
                stage_line = stage_lines[flags[0]] if flags else ""
            elif mm := re.fullmatch(r'AddMacro\(ss, "(\w+)", (.+)\);', s):
                field, feature = parse_macro_arg(mm.group(2), where)
                macros.append(Macro(mm.group(1), field, feature, n))
                steps.append("macro")
            elif s == "ss << m_tfx_source;":
                steps.append("source")
            elif mm := re.search(r"g_vulkan_shader_cache->(\w+)\(ss\.str\(\)\)", s):
                kind = KINDS.get(mm.group(1))
                if kind is None:
                    raise SourceError(f"{where}: shader cache call {mm.group(1)} has no known stage")
            elif s and (steps and steps[-1] != "source" or "ss" in re.findall(r"\b\w+\b", s)
                        and not s.startswith("std::stringstream ss")):
                # Between the header and the shader text every line is understood, or a conditional AddMacro would
                # pass for an unconditional one.
                raise SourceError(f"{where}: {function} statement {s!r} is not understood; teach {Path(__file__).name}")
        order = [k for k, _ in itertools.groupby(steps)]
        if order != ["header", "stage", "macro", "source"] or kind is None:
            raise SourceError(f"{DEVICE_CPP}:{line0}: {function} builds its source as {order} ({kind}); "
                              f"want header, stage, macros, source")
        recipes.append(StageRecipe(kind, function, selector, stage_line, macros))
    if sorted(r.stage for r in recipes) != ["frag", "vert"]:
        raise SourceError(f"want one TFX vertex and one fragment shader builder, found {[r.function for r in recipes]}")
    return sorted(recipes, key=lambda r: r.stage, reverse=True)


def check_shader_path(cpp: str) -> None:
    if 'ReadShaderSource("shaders/vulkan/tfx.glsl")' not in cpp or not TFX.endswith("shaders/vulkan/tfx.glsl"):
        raise SourceError(f"{DEVICE_CPP} no longer reads shaders/vulkan/tfx.glsl into m_tfx_source")


def parse_enums(enums_h: str) -> dict[str, list[int] | SourceError]:
    """Each enum class's member values: `NAME = <integer>`, or `NAME` (the previous value + 1, the first 0). An enum
    with any other member maps to the SourceError, raised only if a selector field uses it."""
    out = {}
    for m in re.finditer(r"enum class (\w+)\s*:\s*\w+\s*\{(.*?)\}", strip_c_comments(enums_h), re.S):
        values, nxt = [], 0
        for member in (x.strip() for x in m.group(2).split(",")):
            if not member:
                continue
            mm = re.fullmatch(r"\w+(?:\s*=\s*((?:0[xX][0-9a-fA-F]+|\d+)[uU]?))?", member)
            if not mm:
                values = SourceError(f"{ENUMS_H}: enum {m.group(1)} member {member!r} is not NAME or NAME = <integer>")
                break
            nxt = int_literal(mm.group(1)) if mm.group(1) else nxt
            values.append(nxt)
            nxt += 1
        out[m.group(1)] = values
    return out


def parse_selector(device_h: str, name: str, enums: dict[str, list[int]]) -> dict[str, list[int]]:
    """Field -> every value it can hold: its enum's values, or 0 .. 2^bits - 1."""
    body, _ = function_body(strip_c_comments(device_h), rf"struct {name}\b")
    inner = re.search(r"\bstruct\s*\{", body)  # the first anonymous struct in the union: the bitfields
    if not inner:
        raise SourceError(f"{DEVICE_H}: {name} has no bitfield struct")
    struct = braced(body, inner.end() - 1)
    aliases = dict(re.findall(r"using (\w+)\s*=\s*GSShader::(\w+);", device_h))
    fields = {}
    for type_, field, bits in re.findall(r"(\w+)\s+(\w+)\s*:\s*(\d+)\s*;", struct):
        width = range(1 << int(bits))
        if type_ in ("u8", "u16", "u32", "u64"):
            fields[field] = list(width)
        elif aliases.get(type_, type_) in enums:
            values = enums[aliases.get(type_, type_)]
            if isinstance(values, SourceError):
                raise values
            fields[field] = [v for v in values if v in width]
        else:
            raise SourceError(f"{DEVICE_H}: {name}.{field} has type {type_}, neither unsigned nor a GSShader enum")
    if not fields:
        raise SourceError(f"{DEVICE_H}: no bitfields in {name}")
    return fields


# ---------------------------------------------------------------------------------------------------------------------
# Reading tfx.glsl: which values of a macro can change the compiled text.

OPS = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
       ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "&": lambda a, b: a & b}
FLIP = {"==": "==", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<=", "&": "&"}


@dataclasses.dataclass
class Uses:
    atoms: set[tuple[str, int]]
    exact: bool = False  # compared with another macro
    value: bool = False  # used as a number in code


TOKEN = re.compile(r"0[xX][0-9a-fA-F]+[uU]?|\d+[uU]?|[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||<<|>>|\S")
INT = re.compile(r"(?:0[xX][0-9a-fA-F]+|\d+)[uU]?")
CMP = {"==", "!=", "<", "<=", ">", ">=", "&"}
# What may stand next to a macro tested for truth, or next to a whole comparison: anything else (arithmetic, a
# function-like macro's parameter, a cast) is a use this script cannot read.
LEFT_OK = {None, "(", "!", "&&", "||"}
RIGHT_OK = {None, ")", "&&", "||"}
CODE_LEFT_OK = LEFT_OK | {"?", ":", ",", "=", "[", "{", "return"}
CODE_RIGHT_OK = RIGHT_OK | {"?", ":", ",", ";", "]", "}"}


def int_literal(token: str) -> int:
    return int(token.rstrip("uU"), 16 if token[:2] in ("0x", "0X") else 10)


def shader_constants(glsl: str) -> dict[str, int]:
    """`#define NAME 3`, `0x10`, `2u`, `(3)`: names standing for one integer."""
    out = {}
    for name, body in re.findall(r"^\s*#\s*define\s+(\w+)[ \t]+(.+?)\s*$", glsl, re.M):
        m = re.fullmatch(r"\(?\s*((?:0[xX][0-9a-fA-F]+|\d+)[uU]?)\s*\)?", body)
        if m:
            out.setdefault(name, int_literal(m.group(1)))
    return out


def macro_uses(glsl: str, macros: set[str]) -> dict[str, Uses]:
    """How tfx.glsl reads each macro: compared with integers, compared with another macro, tested for truth, or (in
    code) used as a number. In a directive anything else is a SourceError: a comparison this script cannot evaluate
    would otherwise merge values that compile differently."""
    text = strip_c_comments(glsl)
    consts = shader_constants(text)
    uses = {m: Uses(set()) for m in macros}

    def operand(toks, k):
        """The integer, constant or macro at toks[k] (a parenthesised one too) and the index past it, else None."""
        if k + 2 < len(toks) and toks[k] == "(" and toks[k + 2] == ")":
            got = operand(toks, k + 1)
            return (got[0], k + 3) if got else None
        if 0 <= k < len(toks):
            t = toks[k]
            if INT.fullmatch(t):
                return int_literal(t), k + 1
            if t in macros:
                return t, k + 1
            if t in consts:
                return consts[t], k + 1
        return None

    def operand_left(toks, k):
        if k - 2 >= 0 and toks[k] == ")" and toks[k - 2] == "(":
            got = operand_left(toks, k - 1)
            return (got[0], k - 2) if got else None
        got = operand(toks, k) if k >= 0 else None
        return (got[0], k) if got else None

    for n, line in enumerate(text.splitlines(), 1):
        toks = TOKEN.findall(line)
        directive = bool(toks) and toks[0] == "#"
        if directive and len(toks) > 1 and toks[1] in ("ifdef", "ifndef", "undef"):
            continue
        for i, t in enumerate(toks):
            if t not in uses:
                continue
            if directive and toks[1:3] == ["define", t] and i == 2:
                continue
            if toks[max(0, i - 2):i] in (["defined", "("],) or (i and toks[i - 1] == "defined"):
                continue
            lo, hi = i, i
            while (lo > 0 and hi + 1 < len(toks) and toks[lo - 1] == "(" and toks[hi + 1] == ")"
                   and (lo < 2 or not re.fullmatch(r"\w+", toks[lo - 2]) or toks[lo - 2] in ("if", "elif", "return"))):
                lo, hi = lo - 1, hi + 1

            def at(k):
                if directive and k <= 1 or k < 0 or k >= len(toks):
                    return None
                return toks[k]
            left_ok, right_ok = (LEFT_OK, RIGHT_OK) if directive else (CODE_LEFT_OK, CODE_RIGHT_OK)
            use = uses[t]
            right, left = at(hi + 1), at(lo - 1)
            if right in CMP and left in left_ok and left != "!":
                got = operand(toks, hi + 2)
                if got and at(got[1]) in right_ok:
                    if isinstance(got[0], str):
                        use.exact = True
                    else:
                        use.atoms.add((right, got[0]))
                    continue
            if left in CMP and right in right_ok:
                got = operand_left(toks, lo - 2)
                if got and at(got[1] - 1) in left_ok and at(got[1] - 1) != "!":
                    if isinstance(got[0], str):
                        use.exact = True
                    else:
                        use.atoms.add((FLIP[left], got[0]))
                    continue
            if left in LEFT_OK and right in RIGHT_OK:
                use.atoms.add(("!=", 0))
                continue
            if directive:
                raise SourceError(f"tfx.glsl:{n}: {t} is read as {line.strip()!r}, which "
                                  f"{Path(__file__).name} cannot evaluate; compare it with an integer or a "
                                  f"#define'd integer, or teach the script")
            use.atoms.add(("!=", 0))
            use.value = True
    return uses


def value_classes(domain: list[int], use: Uses | None) -> list[int]:
    """One value per class of values tfx.glsl cannot tell apart (the smallest), all of them when it compares the
    macro with another macro, plus the largest when it uses the value as a number."""
    if use is None or not domain:
        return domain[:1]
    if use.exact:
        return list(domain)
    atoms = sorted(use.atoms)
    reps = {}
    for v in domain:
        reps.setdefault(tuple(OPS[o](v, c) for o, c in atoms), v)
    out = set(reps.values())
    if use.value:
        out.add(max(domain))
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------------
# Variants.


@dataclasses.dataclass(frozen=True)
class Param:
    name: str  # macro name, or "features.x"
    values: tuple[int, ...]


@dataclasses.dataclass
class Stage:
    recipe: StageRecipe
    params: list[Param]
    constraints: list[Constraint]
    groups: list[tuple[str, ...]]


@dataclasses.dataclass
class Variant:
    stage: str
    values: dict[str, int]  # every Param of the stage
    source: str
    header_lines: int  # lines before the shader text

    def describe(self) -> str:
        feats = " ".join(f"{k[9:]}={v}" for k, v in sorted(self.values.items()) if k.startswith("features."))
        macros = " ".join(f"{k}={v}" for k, v in self.values.items() if not k.startswith("features.") and v)
        return f"{self.stage} [{feats}] {macros or '(all macros 0)'}"


@dataclasses.dataclass
class Sources:
    header: list[HeaderLine]
    macro_format: str  # AddMacro's output, a str.format over name and value
    stages: list[Stage]
    tfx: str


def load(root: Path = ROOT, tfx_path: Path | None = None) -> Sources:
    cpp = strip_c_comments((root / DEVICE_CPP).read_text())
    check_shader_path(cpp)
    header, header_features = parse_header(cpp)
    macro_format = "".join(parse_add_macro(cpp))
    enums = parse_enums((root / ENUMS_H).read_text())
    device_h = (root / DEVICE_H).read_text()
    tfx = (tfx_path or root / TFX).read_text()
    stages = []
    for recipe in parse_stages(cpp):
        fields = parse_selector(device_h, recipe.selector, enums)
        feature_names = set(header_features) | {m.feature for m in recipe.macros if m.feature}
        uses = macro_uses(tfx, {m.name for m in recipe.macros})
        params = []
        for m in recipe.macros:
            if m.feature:
                continue
            if m.field not in fields:
                raise SourceError(f"{DEVICE_CPP}:{m.line}: {m.name} is set from sel.{m.field}, "
                                  f"which {recipe.selector} does not have")
            # One-parameter constraints narrow the domain first, so the classes (and a value-used macro's largest
            # value) come from values the game builds.
            domain = [x for x in fields[m.field]
                      if all(c.ok({m.name: x}) for c in CONSTRAINTS if c.stage == recipe.stage and c.params == (m.name,))]
            params.append(Param(m.name, tuple(value_classes(domain, uses[m.name]))))
        params += [Param(f, (0, 1)) for f in sorted(feature_names)]
        names = {p.name for p in params}
        constraints = [c for c in CONSTRAINTS if c.stage == recipe.stage]
        for c in constraints:
            if not set(c.params) <= names:
                raise SourceError(f"constraint on {c.params} names something {recipe.stage} no longer has")
        stages.append(Stage(recipe, params, constraints, interaction_groups(tfx, header, names)))
    return Sources(header, macro_format, stages, tfx)


def assemble(src: Sources, stage: Stage, values: dict[str, int]) -> tuple[str, int]:
    """The source GetTFX*Shader hands to the shader cache for these selector and feature values."""
    F = {k: bool(v) for k, v in values.items() if k.startswith("features.")}
    parts = [h.text for h in src.header if h.guard is None or eval(h.guard, {"__builtins__": {}}, {"F": F})]
    parts.append(stage.recipe.stage_line)
    for m in stage.recipe.macros:
        parts.append(src.macro_format.format(name=m.name, value=values[m.feature or m.name]))
    prefix = "".join(parts)
    return prefix + src.tfx, prefix.count("\n")


def interaction_groups(glsl: str, header: list[HeaderLine], names: set[str]) -> list[tuple[str, ...]]:
    """For each #if / #elif of tfx.glsl, the parameters it and every block around it read (an #elif or #else branch
    also reads the conditions before it), with the shader's own #defines (SW_BLEND, ZWRITE...) and the header's
    (DISABLE_TEXTURE_BARRIER: the features guarding it) expanded; the groups of three or more."""
    text = strip_c_comments(glsl)
    defines = {}
    for m in re.finditer(r"^\s*#\s*define\s+(\w+)[ \t]+(.+)$", text, re.M):
        defines.setdefault(m.group(1), m.group(2))
    from_header = {}
    for h in header:
        m = re.fullmatch(r"#define (\w+).*\n", h.text)
        if m and h.guard:
            from_header[m.group(1)] = set(re.findall(r'F\["([\w.]+)"\]', h.guard))

    def reads(expr: str, seen: frozenset = frozenset()) -> set[str]:
        out = set()
        for w in re.findall(r"\b[A-Za-z_]\w*\b", expr):
            if w in names:
                out.add(w)
            elif w in from_header:
                out |= from_header[w] & names
            elif w in defines and w not in seen:
                out |= reads(defines[w], seen | {w})
        return out
    stack, groups = [], set()
    for n, line in enumerate(text.splitlines(), 1):
        m = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
        if not m:
            continue
        directive, rest = m.groups()
        if directive == "if":
            stack.append(reads(rest))
        elif directive in ("ifdef", "ifndef"):  # selector macros are always defined: only the header's count
            stack.append(set().union(*(from_header[w] & names for w in re.findall(r"\w+", rest) if w in from_header)))
        elif directive == "elif" and stack:
            stack[-1] = stack[-1] | reads(rest)
        elif directive == "endif" and stack:
            stack.pop()
            continue
        elif directive == "else" and stack:
            continue
        else:
            raise SourceError(f"tfx.glsl:{n}: #{directive} without #if")
        group = set().union(*stack)
        if len(group) >= 3:
            groups.add(tuple(sorted(group)))
    if stack:
        raise SourceError("tfx.glsl: #if without #endif")
    return sorted(groups)


def covering(params: list[Param], groups: list[tuple[str, ...]], constraints: list[Constraint]
             ) -> tuple[list[dict[str, int]], list[tuple], list[tuple]]:
    """Rows (param -> value) holding every value of every param, every pair of values of every two params, and every
    triple of values of every three params in one group; and the tuples left out, those a constraint forbids and those
    the greedy fill could not complete into an allowed row. Greedy and deterministic: a row starts from the first uncovered tuple (triples first) and fills the
    other params, largest domains first, with the value completing the most uncovered tuples (ties: the smaller)."""
    names = [p.name for p in params]
    index = {n: i for i, n in enumerate(names)}
    values = {p.name: p.values for p in params}
    watch: dict[str, list[Constraint]] = {}
    for c in constraints:
        for n in c.params:
            watch.setdefault(n, []).append(c)

    def allowed(row: dict, changed: str) -> bool:
        """No constraint on `changed` is broken by `row`, whatever its params not yet in the row become."""
        for c in watch.get(changed, ()):
            free = [n for n in c.params if n not in row]
            if not any(c.ok({**row, **dict(zip(free, vs))}) for vs in itertools.product(*(values[n] for n in free))):
                return False
        return True

    combos = {(n,) for n in names} | set(itertools.combinations(names, 2))
    for g in groups:
        combos |= set(itertools.combinations(sorted(g, key=index.get), 3))
    seeds = sorted(((c, v) for c in combos for v in itertools.product(*(values[n] for n in c))),
                   key=lambda t: (-len(t[0]), [index[n] for n in t[0]], t[1]))
    uncovered = set(seeds)
    open_vals = {c: set() for c in combos}  # the value tuples of each combo no row holds yet
    for c, vals in seeds:
        open_vals[c].add(vals)

    def cover(tuples):
        for c, vals in tuples:
            uncovered.discard((c, vals))
            open_vals[c].discard(vals)

    fill = sorted(names, key=lambda n: (-len(values[n]), index[n]))
    by_param: dict[str, list[tuple[str, ...]]] = {n: [] for n in names}
    for c in combos:
        for n in c:
            by_param[n].append(c)

    rows, ruled_out, unplaced, next_seed = [], [], [], 0
    while uncovered:
        while seeds[next_seed] not in uncovered:
            next_seed += 1
        seed = seeds[next_seed]
        row, ok = {}, True
        for n, v in zip(*seed):
            row[n] = v
            ok = ok and allowed(row, n)
        if not ok:
            cover([seed])
            ruled_out.append(seed)
            continue
        for n in fill:
            if not ok or n in row:
                continue
            check = [c for c in by_param[n] if open_vals[c] and all(m in row for m in c if m != n)]
            best, best_score = None, -1
            for v in values[n]:
                row[n] = v
                if not allowed(row, n):
                    continue
                score = sum(tuple(row[m] for m in c) in open_vals[c] for c in check)
                if score > best_score:
                    best, best_score = v, score
            if best is None:
                ok = False
                del row[n]
            else:
                row[n] = best
        if not ok:
            cover([seed])
            unplaced.append(seed)
            continue
        cover([(c, vals) for c in combos if open_vals[c] for vals in [tuple(row[m] for m in c)] if vals in open_vals[c]])
        rows.append({n: row[n] for n in names})
    return rows, ruled_out, unplaced


@dataclasses.dataclass
class Stats:
    rows: int
    ruled_out: list[tuple]
    unplaced: list[tuple]
    duplicates: int


def variants(src: Sources) -> tuple[list[Variant], dict[str, Stats]]:
    """Every stage's variants (one per distinct source), and per stage how they were made."""
    out, stats = [], {}
    for stage in src.stages:
        rows, ruled_out, unplaced = covering(stage.params, stage.groups, stage.constraints)
        seen, dupes = set(), 0
        for row in rows:
            text, hdr = assemble(src, stage, row)
            if text in seen:
                dupes += 1
                continue
            seen.add(text)
            out.append(Variant(stage.recipe.stage, row, text, hdr))
        stats[stage.recipe.stage] = Stats(len(rows), ruled_out, unplaced, dupes)
    return out, stats


# ---------------------------------------------------------------------------------------------------------------------
# Compiling.


def compile_one(compiler: list[str], v: Variant) -> tuple[Variant, int, str]:
    r = subprocess.run(compiler + [v.stage], input=v.source.encode(), capture_output=True)
    return v, r.returncode, (r.stderr + r.stdout).decode(errors="replace")


def tfx_lines(message: str, header_lines: int) -> str:
    """`source:123:` (a line of the assembled source) also as the tfx.glsl line."""
    return re.sub(r"\bsource:(\d+):", lambda m: f"source:{m.group(1)} (tfx.glsl:{int(m.group(1)) - header_lines}):",
                  message)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compiler", default="tfxcc", help="tfxcc, or any command taking vert|frag, the source on stdin, exit 0 = compiled")
    ap.add_argument("--tfx", type=Path, help=f"the shader to check instead of {TFX} (a scratch copy)")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--jobs", type=int, default=0, help="compiles at once (0: one per CPU)")
    ap.add_argument("--list", action="store_true", help="print the parameters and variants, compile nothing")
    ap.add_argument("--dump", type=Path, help="write every variant's source here")
    args = ap.parse_args(argv)
    args.jobs = args.jobs if args.jobs > 0 else os.cpu_count() or 4

    start = time.monotonic()
    try:
        src = load(args.root, args.tfx)
        for c in CONSTRAINTS:
            cite(c, args.root)
    except SourceError as e:
        print(f"check_tfx_variants: {e}", file=sys.stderr)
        return 2
    vs, stats = variants(src)

    if args.list:
        for stage in src.stages:
            print(f"{stage.recipe.stage}: {len(stage.params)} parameters")
            for p in stage.params:
                print(f"  {p.name} {list(p.values)}")
            for g in stage.groups:
                print(f"  group {' '.join(g)}")
            for c in stage.constraints:
                print(f"  constraint {c.params}: {c.why} ({cite(c, args.root)})")
        for i, v in enumerate(vs):
            print(f"{i:4} {v.describe()}")
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        for i, v in enumerate(vs):
            (args.dump / f"{i:04}.{v.stage}.glsl").write_text(v.source)
    if args.list:
        return 0

    compiler = shlex.split(args.compiler)
    failed = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for v, rc, msg in pool.map(lambda v: compile_one(compiler, v), vs):
            if rc != 0:
                failed.append((v, rc, msg))
    for v, rc, msg in failed:
        print(f"FAIL {v.describe()}")
        for line in tfx_lines(msg, v.header_lines).strip().splitlines()[:12]:
            print(f"    {line}")
    unplaced = [(name, t) for name, st in stats.items() for t in st.unplaced]
    for name, (combo, vals) in unplaced:
        print(f"UNPLACED {name}: no variant holds {' '.join(f'{n}={v}' for n, v in zip(combo, vals))}; "
              f"a constraint missing from {Path(__file__).name}, or the fill needs to backtrack")
    per_stage = ", ".join(f"{name} {sum(v.stage == name for v in vs)}" for name in stats)
    print(f"tfx variants: {len(vs)} ({per_stage}), {len(failed)} failed; every pair of "
          f"{' / '.join(f'{len(st.params)} {st.recipe.stage}' for st in src.stages)} parameters and every triple in "
          f"{sum(len(st.groups) for st in src.stages)} #if groups, "
          f"{sum(len(st.ruled_out) for st in stats.values())} combinations ruled out by {len(CONSTRAINTS)} constraints; "
          f"{args.jobs} jobs, {time.monotonic() - start:.1f}s")
    return 1 if failed or unplaced else 0


if __name__ == "__main__":
    sys.exit(main())
