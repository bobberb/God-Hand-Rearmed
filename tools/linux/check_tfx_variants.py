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
    compiled. A macro compared with another macro keeps all its values; one used in code other than in a comparison
    (int(M), f(M), if (M)) also keeps its largest.

Coverage rule, per stage (vert, frag), over the parameters (selector macros and device features):
  - every value class of every parameter, and every pair of classes of every two parameters;
  - every triple of classes of every three parameters one block of tfx.glsl is conditioned on (its #if / #elif and
    all those around it, with the shader's own #defines like SW_BLEND or ZWRITE and the header's like
    DISABLE_TEXTURE_BARRIER expanded to what they read);
  - every line of tfx.glsl a game build can compile, compiled by some variant: each block (#if stack) no variant
    compiles gets a search for parameter values that compile it; none found proves no game build does;
  - for every use of a name whose declarations in reach (in its function, its function's parameters, the globals,
    the shader's #defines) are all under some condition not around the use: a variant compiling the use and none
    of them, if the constraints allow one. Uses through a #define body count, under that definition's #if.
  Both searches are depth first over the #if conditions (evaluated, three-valued) and prove it when they find
  nothing; one past its budget is an error. Declarations are read by type (GLSL's, and the structs tfx.glsl
  declares), with declarator lists and `type[N]`. Not told apart: block scopes inside a function (a name declared
  in an inner { } counts for the whole function), names made by token pasting or function-like macro arguments, and
  interface-block instance names; a variant this adds for nothing compiles, one it misses is a gap.
All of it in a greedy covering array, deterministic. Combinations no game build can contain are left out by
CONSTRAINTS below, each citing the game code that rules it out; one the greedy fill could not place in any variant is
an error. Variants whose assembled source is the same text are compiled once.

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
GSREGS = "thirdparty/pcsx2/pcsx2/GS/GSRegs.h"


class SourceError(Exception):
    """The C++ or the shader has a shape this script does not know; fix the script, never skip."""


# ---------------------------------------------------------------------------------------------------------------------
# Constraints: combinations the game never builds. Each cites the code that rules the combination out: (path, regex,
# n) must match exactly n lines of path (test_check_tfx_variants.py checks it), so the file:line printed stays right
# when the file moves and a citation goes stale loudly.


@dataclasses.dataclass(frozen=True)
class Constraint:
    stage: str
    params: tuple[str, ...]
    ok: Callable[[dict], bool]
    why: str
    cites: tuple[tuple[str, str, int], ...]


CONSTRAINTS = [
    Constraint("frag", ("PS_ROV_COLOR", "PS_ROV_DEPTH", "features.rov"),
               lambda v: not (v["PS_ROV_COLOR"] or v["PS_ROV_DEPTH"]) or v["features.rov"],
               "ROVs only on a device with ROV support (the header's interlock extensions come with it)",
               ((RENDERER_HW, r"if \(!\(GSConfig\.HWROV && features\.rov\)\)", 1),)),
    Constraint("frag", ("PS_ROV_COLOR", "PS_NO_COLOR1"),
               lambda v: not v["PS_ROV_COLOR"] or v["PS_NO_COLOR1"],
               "colour ROV blends in the shader and drops the dual-source output; with blending off the pipeline "
               "drops it too",
               ((RENDERER_HW, r"// We should never need dual source with SW blend", 1),
                (DEVICE_CPP, r"pps\.no_color1 = true;", 1))),
    Constraint("frag", ("PS_NO_COLOR", "PS_NO_COLOR1"),
               lambda v: not v["PS_NO_COLOR"] or v["PS_NO_COLOR1"],
               "no colour output means no second colour output either",
               ((DEVICE_H, r"no_color = no_color1 = 1;", 1),)),
    Constraint("frag", ("PS_DATE",),
               lambda v: v["PS_DATE"] in (0, 1, 2, 3, 5, 6),
               "DATE is 1 + DATM (PrimID), 5 + DATM (barrier, stencil one, ROV) or 3 (after the PrimID pass); DATM "
               "is one bit",
               ((RENDERER_HW, r"m_conf\.ps\.date = 1 \+ m_cached_ctx\.TEST\.DATM;", 1),
                (RENDERER_HW, r"m_conf\.ps\.date = 5 \+ m_cached_ctx\.TEST\.DATM;", 3),
                (DEVICE_CPP, r"^\tconfig\.ps\.date = 3;", 1))),
    Constraint("frag", ("PS_ANISOTROPIC_FILTERING",),
               lambda v: v["PS_ANISOTROPIC_FILTERING"] <= 16,
               "sw_aniso is GSConfig.MaxAnisotropy or 0, and the launcher's option is clamped to 0..16",
               ((RENDERER_HW, r"m_conf\.ps\.sw_aniso = anisotropic \? GSConfig\.MaxAnisotropy : 0;", 1),
                (BRIDGE, r"options\.MaxAnisotropy = static_cast<u8>\(std::clamp\(value, 0, 16\)\);", 1))),
    Constraint("frag", ("PS_DATE", "PS_NO_COLOR"),
               lambda v: v["PS_DATE"] not in (1, 2) or not v["PS_NO_COLOR"],
               "DATE 1 and 2 are the PrimID pass, which writes colour",
               ((DEVICE_CPP, r"pipe\.ps\.no_color = false;", 1),)),
    # Selectors wider than their values (tfx is 3 bits, blend_hw 3, blend_mix 2). The other wide ones copy a register
    # field of their own width (ALPHA.A..D, TEST.ZTST, SCANMSK, CLAMP, psm.fmt), use every value (pal_fmt,
    # depth_fmt, process_*, dither), or their spare values already share class 0 (channel 7).
    Constraint("frag", ("PS_TFX",),
               lambda v: v["PS_TFX"] <= 4,
               "tfx is TFX_DECAL, TEX0.TFX (2 bits) or 4 (TFX_NONE)",
               ((RENDERER_HW, r"m_conf\.ps\.tfx = m_cached_ctx\.TEX0\.TFX;", 1),
                (RENDERER_HW, r"m_conf\.ps\.tfx = 4;", 3),
                (GSREGS, r"TFX_NONE\s+= 4,", 1))),
    Constraint("frag", ("PS_BLEND_HW",),
               lambda v: v["PS_BLEND_HW"] <= 6,
               "blend_hw is a HWBlendType (at most INV_SRC_DST_BLEND_HALF = 6), here or through blend_multi_pass",
               ((DEVICE_H, r"INV_SRC_DST_BLEND_HALF\s+= 6,", 1),
                (DEVICE_CPP, r"pipe\.ps\.blend_hw = config\.blend_multi_pass\.blend_hw;", 1))),
    Constraint("frag", ("PS_BLEND_MIX",),
               lambda v: v["PS_BLEND_MIX"] <= 2,
               "blend_mix is 1 or 2 (REV_SUBTRACT), or off",
               ((RENDERER_HW, r"m_conf\.ps\.blend_mix = \(blend\.op == GSDevice::OP_REV_SUBTRACT\) \? 2 : 1;", 1),
                (RENDERER_HW, r"m_conf\.ps\.blend_mix = false;", 1))),
    Constraint("frag", ("PS_DATE", "PS_NO_COLOR1", "PS_BLEND_A", "PS_BLEND_B", "PS_BLEND_C", "PS_BLEND_D"),
               lambda v: v["PS_DATE"] not in (1, 2) or (v["PS_NO_COLOR1"] and not (
                   v["PS_BLEND_A"] or v["PS_BLEND_B"] or v["PS_BLEND_C"] or v["PS_BLEND_D"])),
               "the PrimID pass (DATE 1 and 2) has no dual-source output and no shader blending",
               ((DEVICE_CPP, r"^\tpipe\.ps\.no_color1 = true;", 1),
                (DEVICE_CPP, r"pipe\.ps\.blend_a = pipe\.ps\.blend_b = pipe\.ps\.blend_c = pipe\.ps\.blend_d = false;",
                 1))),
]


def cite(c: Constraint, root: Path = ROOT) -> str:
    """Every line the constraint cites, as path:line."""
    out = []
    for path, anchor, want in c.cites:
        lines = (root / path).read_text().splitlines()
        hits = [i + 1 for i, line in enumerate(lines) if re.search(anchor, line)]
        if len(hits) != want:
            raise SourceError(f"constraint citation {anchor!r} matches {len(hits)} lines of {path}, want {want}")
        out += [f"{path}:{n}" for n in hits]
    return ", ".join(out)


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
    """AddShaderHeader: `ss << "...";` lines, each under the `if (...)` guarding it (one statement or a braced
    block)."""
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
    body, line0 = function_body(
        cpp, r"static void AddShaderStageMacro\(std::stringstream& ss, bool vs, bool gs, bool fs\)")
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


def parse_stages(cpp: str, stage_lines: dict[str, str] | None = None) -> list[StageRecipe]:
    """Each GSDeviceVK::GetTFX*Shader: header, stage macro, AddMacro lines in order, then the shader text."""
    stage_lines = stage_lines or parse_stage_macros(cpp)
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
    if 'ReadShaderSource("shaders/vulkan/tfx.glsl")' not in cpp:
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
            # A truth test only in a directive: in code, int(M), f(M) or if (M) all see the number.
            if directive and left in LEFT_OK and right in RIGHT_OK:
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
    # Uses of a name compiled without its declaration: (name, tfx.glsl line, the parameter values doing it); and the
    # ones whose search gave up.
    missing: list[tuple[str, int, dict]] = dataclasses.field(default_factory=list)
    gave_up: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    cond: "Conditions | None" = None


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
    warnings: list[str] = dataclasses.field(default_factory=list)


def check_names(glsl: str, header: list[HeaderLine], recipes: list[StageRecipe]) -> None:
    """A PS_ / VS_ / GS_ name tfx.glsl reads that neither GSDeviceVK.cpp nor tfx.glsl itself defines is a
    SourceError: the preprocessor reads it as 0 without a word, so it is almost always a misspelling."""
    text = strip_c_comments(glsl)
    defined = set(re.findall(r"^\s*#\s*define\s+(\w+)", text, re.M))
    defined |= {m.name for r in recipes for m in r.macros}
    defined |= {m.group(1) for h in header for m in [re.match(r"#define (\w+)", h.text)] if m}
    for n, line in enumerate(text.splitlines(), 1):
        if re.match(r"\s*#\s*(define|undef)\b", line):
            continue
        for name in re.findall(r"\b((?:PS|VS|GS)_[A-Z0-9_]+)\b", line):
            if name not in defined:
                raise SourceError(f"tfx.glsl:{n}: {name} is read but nothing defines it (GSDeviceVK.cpp's AddMacro "
                                  f"or a #define in tfx.glsl); a misspelling?")


def load(root: Path = ROOT, tfx_path: Path | None = None) -> Sources:
    cpp = strip_c_comments((root / DEVICE_CPP).read_text())
    check_shader_path(cpp)
    header, header_features = parse_header(cpp)
    macro_format = "".join(parse_add_macro(cpp))
    enums = parse_enums((root / ENUMS_H).read_text())
    device_h = (root / DEVICE_H).read_text()
    tfx = (tfx_path or root / TFX).read_text()
    stages = []
    stage_lines = parse_stage_macros(cpp)
    recipes = parse_stages(cpp, stage_lines)
    warnings: list[str] = []
    check_names(tfx, header, recipes)
    for recipe in recipes:
        fields = parse_selector(device_h, recipe.selector, enums)
        feature_names = set(header_features) | {m.feature for m in recipe.macros if m.feature}
        uses = macro_uses(tfx, {m.name for m in recipe.macros})
        for m in recipe.macros:
            u = uses[m.name]
            if not (u.atoms or u.exact or u.value):
                # Not an error: a selector another backend reads, or one kept for later, is harmless here.
                warnings.append(f"{DEVICE_CPP}:{m.line}: {recipe.function} adds {m.name}, which tfx.glsl never reads")
        params = []
        for m in recipe.macros:
            if m.feature:
                continue
            if m.field not in fields:
                raise SourceError(f"{DEVICE_CPP}:{m.line}: {m.name} is set from sel.{m.field}, "
                                  f"which {recipe.selector} does not have")
            # One-parameter constraints narrow the domain first, so the classes (and a value-used macro's largest
            # value) come from values the game builds.
            own = [c for c in CONSTRAINTS if c.stage == recipe.stage and c.params == (m.name,)]
            domain = [x for x in fields[m.field] if all(c.ok({m.name: x}) for c in own)]
            params.append(Param(m.name, tuple(value_classes(domain, uses[m.name]))))
        params += [Param(f, (0, 1)) for f in sorted(feature_names)]
        names = {p.name for p in params}
        constraints = [c for c in CONSTRAINTS if c.stage == recipe.stage]
        for c in constraints:
            if not set(c.params) <= names:
                raise SourceError(f"constraint on {c.params} names something {recipe.stage} no longer has")
        own = re.fullmatch(r"#define (\w+) .*\n", recipe.stage_line).group(1)
        others = {re.fullmatch(r"#define (\w+) .*\n", line).group(1) for line in stage_lines.values()} - {own}
        aliases = {m.name: m.feature for m in recipe.macros if m.feature}
        cond = Conditions(tfx, header, own, others, names, aliases)
        stage = Stage(recipe, params, constraints, interaction_groups(cond), cond=cond)
        allowed = checker(params, constraints)
        for name, line, use, decls in use_without_declaration(cond):
            env, finished = find_missing_declaration(cond, use, decls, params, constraints, allowed)
            if env:
                stage.missing.append((name, line, env))
            elif not finished:
                stage.gave_up.append((name, line))
        stages.append(stage)
    return Sources(header, macro_format, stages, tfx, warnings)


def assemble(src: Sources, stage: Stage, values: dict[str, int]) -> tuple[str, int]:
    """The source GetTFX*Shader hands to the shader cache for these selector and feature values."""
    F = {k: bool(v) for k, v in values.items() if k.startswith("features.")}
    parts = [h.text for h in src.header if h.guard is None or eval(h.guard, {"__builtins__": {}}, {"F": F})]
    parts.append(stage.recipe.stage_line)
    for m in stage.recipe.macros:
        parts.append(src.macro_format.format(name=m.name, value=values[m.feature or m.name]))
    prefix = "".join(parts)
    return prefix + src.tfx, prefix.count("\n")


def interaction_groups(cond: "Conditions") -> list[tuple[str, ...]]:
    """For each block of tfx.glsl, the parameters it and every block around it read (an #elif or #else branch also
    reads the conditions before it), with the shader's #defines, the header's and the feature-set AddMacro names
    resolved to what they read (Conditions); the groups of three or more."""
    groups = set()
    for stack in {stack for _, stack, _ in cond.lines}:
        group = set().union(set(), *(cond.reads(c) for c in stack))
        if len(group) >= 3:
            groups.add(tuple(sorted(group)))
    return sorted(groups)


# ---------------------------------------------------------------------------------------------------------------------
# Which code a variant compiles: tfx.glsl's #if expressions evaluated, three-valued (None: depends on a parameter not
# chosen yet), so a search can find a variant that compiles a use of a name without any of its declarations.

PRECEDENCE = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", "<=", ">", ">="), ("<<", ">>"),
              ("+", "-"), ("*", "/", "%")]


def parse_condition(tokens: list[str], where: str):
    """A preprocessor expression as a tree: ("num", n), ("id", name), ("defined", name), ("un", op, x),
    ("bin", op, x, y)."""
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take(want=None):
        nonlocal pos
        t = peek()
        if t is None or want is not None and t != want:
            raise SourceError(f"{where}: cannot read the condition {' '.join(tokens)!r}")
        pos += 1
        return t

    def unary():
        t = take()
        if t in ("!", "~", "-", "+"):
            return ("un", t, unary())
        if t == "(":
            x = binary(0)
            take(")")
            return x
        if t == "defined":
            if peek() == "(":
                take("(")
                name = take()
                take(")")
            else:
                name = take()
            return ("defined", name)
        if INT.fullmatch(t):
            return ("num", int_literal(t))
        if re.fullmatch(r"[A-Za-z_]\w*", t) and peek() != "(":
            return ("id", t)
        raise SourceError(f"{where}: cannot read the condition {' '.join(tokens)!r}")

    def binary(level):
        if level == len(PRECEDENCE):
            return unary()
        x = binary(level + 1)
        while peek() in PRECEDENCE[level]:
            op = take()
            x = ("bin", op, x, binary(level + 1))
        return x

    tree = binary(0)
    if pos != len(tokens):
        raise SourceError(f"{where}: cannot read the condition {' '.join(tokens)!r}")
    return tree


ARITH = {"|": lambda a, b: a | b, "^": lambda a, b: a ^ b, "&": lambda a, b: a & b, "==": lambda a, b: int(a == b),
         "!=": lambda a, b: int(a != b), "<": lambda a, b: int(a < b), "<=": lambda a, b: int(a <= b),
         ">": lambda a, b: int(a > b), ">=": lambda a, b: int(a >= b), "<<": lambda a, b: a << b,
         ">>": lambda a, b: a >> b, "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
         "/": lambda a, b: int(a / b) if b else None, "%": lambda a, b: a % b if b else None}


class Conditions:
    """tfx.glsl's #if / #elif / #else structure for one stage: each line's stack of block conditions, resolved once
    (the shader's own macros inlined, the header's as guards over device features, anything else a number), and a
    three-valued evaluator of them: 1, 0, or None while it depends on parameters not set yet."""

    def __init__(self, glsl: str, header: list[HeaderLine], stage_macro: str, other_stage_macros: set[str],
                 names: set[str], aliases: dict[str, str] | None = None):
        self.names = names
        # AddMacro names set from a device feature (VS_PROVOKING_VERTEX_LAST): read as that feature's parameter.
        self.aliases = aliases or {}
        self.stage_macro, self.other_stage_macros = stage_macro, other_stage_macros
        text = strip_c_comments(glsl)
        self.defines: dict[str, set[str]] = {}
        for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_]\w*)(\(?)[ \t]*(.*?)\s*$", text, re.M):
            self.defines.setdefault(m.group(1), set()).add("(" if m.group(2) else m.group(3))
        # Header macros: name -> (as defined(name), as name): a feature guard over 1 and over the macro's value.
        self.header: dict[str, tuple[tuple, tuple]] = {}
        for h in header:
            m = re.fullmatch(r"#define (\w+)(?:[ \t]+(.*?))?\s*", h.text)
            if m:
                value = (("num", int_literal(m.group(2))) if m.group(2) and INT.fullmatch(m.group(2)) else
                         ("unreadable", m.group(1), m.group(2) or ""))
                if not h.guard:
                    self.header[m.group(1)] = (("num", 1), value)
                    continue
                guard = (compile(h.guard.strip(), "guard", "eval"),
                         tuple(sorted(set(re.findall(r'F\["([\w.]+)"\]', h.guard)))))
                self.header[m.group(1)] = (("guard", *guard, ("num", 1)), ("guard", *guard, value))
        self._bodies: dict[str, tuple] = {}
        self._reads: dict[int, frozenset] = {}
        self.lines: list[tuple[int, tuple, str]] = []  # (line number, condition stack, line)
        stack: list[tuple[tuple, list]] = []  # (this branch's condition, the conditions before it in its chain)
        for n, line in enumerate(text.splitlines(), 1):
            toks = TOKEN.findall(line)
            where = f"tfx.glsl:{n}"
            if toks[:1] == ["#"] and len(toks) > 1 and toks[1] in ("if", "ifdef", "ifndef", "elif", "else", "endif"):
                d = toks[1]
                if d in ("if", "ifdef", "ifndef"):
                    cond = (self.resolve(parse_condition(toks[2:], where)) if d == "if" else
                            self.resolve(("defined", toks[2])) if d == "ifdef" else
                            ("un", "!", self.resolve(("defined", toks[2]))))
                    stack.append((cond, [cond]))
                elif not stack:
                    raise SourceError(f"{where}: #{d} without #if")
                elif d == "endif":
                    stack.pop()
                else:
                    before = stack[-1][1]
                    none_before = before[0]
                    for c in before[1:]:
                        none_before = ("bin", "||", none_before, c)
                    none_before = ("un", "!", none_before)
                    if d == "elif":
                        cond = self.resolve(parse_condition(toks[2:], where))
                        stack[-1] = (("bin", "&&", none_before, cond), before + [cond])
                    else:
                        stack[-1] = (none_before, before)
                continue
            self.lines.append((n, tuple(c for c, _ in stack), line))
        if stack:
            raise SourceError("tfx.glsl: #if without #endif")

    def resolve(self, node, seen: frozenset = frozenset()) -> tuple:
        """`node` with every name replaced: a parameter stays ("id", name); a shader macro becomes its body, a header
        macro its guard, the stage's macro 1, anything else 0 (as the preprocessor reads an unknown name)."""
        kind = node[0]
        if kind == "num":
            return node
        if kind == "defined":
            name = node[1]
            if name in self.names or name in self.aliases:  # the game defines every AddMacro name
                return ("num", 1)
            if name in self.header:
                return self.header[name][0]
            return ("num", int(name == self.stage_macro))
        if kind == "id":
            name = node[1]
            if name in self.names:
                return node
            if name in self.aliases:
                return ("id", self.aliases[name])
            if name in self.header:
                value = self.header[name][1]
                if value[0] == "unreadable" or value[0] == "guard" and value[3][0] == "unreadable":
                    raise SourceError(f"tfx.glsl reads the header's {name} as a number, but the header defines it as "
                                      f"{(value[2] if value[0] == 'unreadable' else value[3][2])!r}")
                return value
            if name == self.stage_macro:
                return ("num", 1)
            if name in self.defines and name not in seen and name not in self.other_stage_macros:
                if name not in self._bodies:
                    bodies = self.defines[name]
                    if len(bodies) != 1 or "(" in bodies:
                        raise SourceError(f"tfx.glsl: {name} is used in a condition but has {len(bodies)} "
                                          f"definitions or takes arguments; teach {Path(__file__).name}")
                    tree = parse_condition(TOKEN.findall(next(iter(bodies))), f"tfx.glsl: #define {name}")
                    self._bodies[name] = self.resolve(tree, seen | {name})
                return self._bodies[name]
            return ("num", 0)
        return (kind, node[1], *(self.resolve(x, seen) for x in node[2:]))

    def value(self, node, env: dict, memo: dict):
        """The integer `node` is with the parameters in env, None if that depends on others. memo: {} per env."""
        key = id(node)
        if key in memo:
            return memo[key]
        kind = node[0]
        if kind == "num":
            out = node[1]
        elif kind == "id":
            out = env.get(node[1])
        elif kind == "guard":  # ("guard", code, features, value): value where the guard holds, else 0
            out = (None if any(f not in env for f in node[2]) else
                   self.value(node[3], env, memo) if eval(node[1], {"__builtins__": {}}, {"F": env}) else 0)
        elif kind == "un":
            x = self.value(node[2], env, memo)
            out = None if x is None else {"!": int(not x), "~": ~x, "-": -x, "+": x}[node[1]]
        else:
            op, a = node[1], self.value(node[2], env, memo)
            if op == "&&":
                b = 0 if a == 0 else self.value(node[3], env, memo)
                out = 0 if a == 0 or b == 0 else None if a is None or b is None else 1
            elif op == "||":
                b = 1 if a not in (0, None) else self.value(node[3], env, memo)
                out = 1 if a not in (0, None) or b not in (0, None) else None if a is None or b is None else 0
            else:
                b = self.value(node[3], env, memo)
                out = None if a is None or b is None else ARITH[node[1]](a, b)
        memo[key] = out
        return out

    def reads(self, node) -> frozenset:
        """The parameters `node` can depend on."""
        key = id(node)
        if key not in self._reads:
            kind = node[0]
            self._reads[key] = (frozenset() if kind == "num" else frozenset({node[1]}) if kind == "id" else
                                frozenset(node[2]) if kind == "guard" else
                                frozenset().union(*(self.reads(x) for x in node[2:])))
        return self._reads[key]

    def open_reads(self, node, env: dict, memo: dict) -> set[str]:
        """The unset parameters that can still change `node`: those of its undecided parts only."""
        if self.value(node, env, memo) is not None:
            return set()
        if node[0] in ("id", "guard"):
            return set(self.reads(node)) - env.keys()
        return set().union(*(self.open_reads(x, env, memo) for x in node[2:]))

    def stack_value(self, stack: tuple, env: dict, memo: dict | None = None):
        """1 if every block in the stack is compiled, 0 if one is not, None if that depends on unset parameters."""
        memo = {} if memo is None else memo
        out = 1
        for cond in stack:
            v = self.value(cond, env, memo)
            if v == 0:
                return 0
            if v is None:
                out = None
        return out


def _glsl_types() -> set[str]:
    out = set("void bool int uint float double atomic_uint".split())
    for n in "234":
        out |= {f"{p}vec{n}" for p in ("", "i", "u", "b", "d")}
        out |= {f"mat{n}", f"dmat{n}"} | {f"{d}mat{n}x{m}" for d in ("", "d") for m in "234"}
    dims = ("1D 2D 3D Cube 2DRect 1DArray 2DArray CubeArray Buffer 2DMS 2DMSArray").split()
    for p in ("", "i", "u"):
        out |= {f"{p}{kind}{d}" for kind in ("sampler", "texture", "image") for d in dims}
        out |= {f"{p}subpassInput", f"{p}subpassInputMS"}
    out |= {f"sampler{d}Shadow" for d in ("1D", "2D", "Cube", "2DRect", "1DArray", "2DArray", "CubeArray")}
    return out | {"sampler", "samplerShadow"}


GLSL_TYPES = _glsl_types()


def declarations_and_uses(cond: "Conditions") -> tuple[dict[str, list[tuple[str | None, tuple]]],
                                                       list[tuple[str, int, str | None, tuple]]]:
    """Every declaration in tfx.glsl as name -> [(scope, stack)] and every use as (name, line, scope, stack), scope
    the enclosing function (None: global). A declaration is a GLSL or struct type (`type[N]` too) followed by its
    declarators (`float a = 0.0, b;` declares both); a function's parameters are in its scope; a #define is global.
    A use of one of the shader's #defines also uses every name in each of its bodies, under the use's #if stack and
    that definition's (so a name only one definition reads is used only where that one is compiled). Lines this
    stage never compiles are left out of both, but count for the braces."""
    text_types = GLSL_TYPES | set(re.findall(r"\bstruct\s+([A-Za-z_]\w*)", "\n".join(x for _, _, x in cond.lines)))
    bodies: dict[str, list[tuple[tuple, set[str]]]] = {}
    for _, stack, line in cond.lines:
        m = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)(?:\([^)]*\))?(.*)", line)
        if m and cond.stack_value(stack, {}) != 0:
            bodies.setdefault(m.group(1), []).append(
                (stack, set(re.findall(r"(?<![.\w])([A-Za-z_]\w*)", m.group(2)))))

    def expand(name, stack, seen=frozenset()):
        """(name, stack) for the name and, through every body of a #define, each name it reads."""
        yield name, stack
        if name not in seen:
            for where, names in bodies.get(name, ()):
                for inner in names:
                    yield from expand(inner, stack + where, seen | {name})

    decls: dict[str, list[tuple[str | None, tuple]]] = {}
    uses = []
    depth, parens, function, pending = 0, 0, None, None
    for n, stack, line in cond.lines:
        live = cond.stack_value(stack, {}) != 0
        m = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)", line)
        if m:
            if live and m.group(1) not in cond.names:
                decls.setdefault(m.group(1), []).append((None, stack))
            continue
        if line.lstrip().startswith("#"):
            continue
        toks = TOKEN.findall(line)
        declared_at = set()
        for i, t in enumerate(toks):
            if t in text_types and (i == 0 or toks[i - 1] != "."):
                j = i + 1
                if j < len(toks) and toks[j] == "[":  # float[2] name
                    while j < len(toks) and toks[j] != "]":
                        j += 1
                    j += 1
                while j < len(toks) and re.fullmatch(r"[A-Za-z_]\w*", toks[j]) and toks[j] not in text_types:
                    name, j = toks[j], j + 1
                    is_function = j < len(toks) and toks[j] == "(" and depth == 0 and parens == 0
                    # In a body: the function's. In the parentheses after a function's name: its parameter.
                    # Anything else at the top (a global, a function's own name) is global.
                    scope = function if depth else pending if parens and pending else None
                    if live and name not in cond.names:
                        decls.setdefault(name, []).append((scope, stack))
                    declared_at.add(j - 1)
                    if is_function:
                        pending = name
                        break
                    nest = 0  # past [N] and = initialiser, to a ',' starting the next declarator or the end
                    while j < len(toks) and (nest or toks[j] not in (",", ";", ")")):
                        nest += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(toks[j], 0)
                        j += 1
                    if j < len(toks) and toks[j] == "," and not (depth == 0 and parens):
                        j += 1
                        continue
                    break
            if t == "(" and depth == 0:
                parens += 1
            elif t == ")" and depth == 0:
                parens -= 1
            elif t == "{":
                if depth == 0:
                    function, pending = pending, None
                depth += 1
            elif t == "}":
                depth -= 1
                if depth == 0:
                    function = None
                if depth < 0:
                    raise SourceError(f"tfx.glsl:{n}: '}}' without '{{'")
            elif t == ";" and depth == 0 and parens == 0:
                pending = None
        if live:
            for i, t in enumerate(toks):
                if re.fullmatch(r"[A-Za-z_]\w*", t) and i not in declared_at and (i == 0 or toks[i - 1] != "."):
                    for name, at in expand(t, stack):
                        uses.append((name, n, function, at))
    if depth:
        raise SourceError("tfx.glsl: unbalanced braces")
    return decls, uses


def use_without_declaration(cond: "Conditions") -> list[tuple[str, int, tuple, list[tuple]]]:
    """(name, line, use stack, declaration stacks) for each use of a name tfx.glsl declares where no declaration in
    reach (the function's own, its parameters, the globals) is sure to be compiled with the use: none of their
    stacks is a prefix of the use's."""
    decls, uses = declarations_and_uses(cond)
    out, seen = [], set()
    for name, n, scope, stack in uses:
        if (name, scope, stack) in seen or name not in decls:
            continue
        seen.add((name, scope, stack))
        reach = sorted({st for sc, st in decls[name] if sc is None or sc == scope}, key=len)
        if reach and not any(stack[:len(d)] == d for d in reach):
            out.append((name, n, stack, reach))
    return out


def find_missing_declaration(cond: Conditions, use: tuple, decls: list[tuple], params: list[Param],
                             constraints: list[Constraint], allowed: Callable[[dict, str], bool],
                             budget: int = 20000) -> tuple[dict | None, bool]:
    """Parameter values (only those the conditions read) under which every block of the use compiles and every
    declaration has a block that does not, allowed by the constraints, or None; and whether the search finished
    (False: it ran past its budget, so None proves nothing).

    Depth first over the parameters the declarations read (those the use reads too first). The parameters only the use
    reads, outside every constraint, cannot change a declaration: they are searched last, once per way of ruling all
    the declarations out, for any values compiling the use."""
    frames = sorted({c for s in [use, *decls] for c in s}, key=repr)
    reads = {c: cond.reads(c) for c in frames}
    use_reads = set().union(set(), *(reads[c] for c in use))
    decl_reads = set().union(set(), *(reads[c] for d in decls for c in d))
    tied = {n for c in constraints for n in c.params}
    use_only = use_reads - decl_reads - tied
    by_name = {p.name: p for p in params}
    steps = 0

    def branch(env, name, then):
        for v in by_name[name].values:
            env[name] = v
            if allowed(env, name):
                got = then(env)
                if got is not None:
                    return got
            del env[name]
        return None

    def compile_use(env):
        """Values of the use-only parameters compiling every block of the use."""
        nonlocal steps
        steps += 1
        if steps > budget:
            return None
        memo: dict = {}
        u = [cond.value(c, env, memo) for c in use]
        if 0 in u:
            return None
        if all(v == 1 for v in u):
            return dict(env)
        open_reads = sorted(set().union(*(cond.open_reads(c, env, memo) for c, v in zip(use, u) if v is None)))
        return branch(env, open_reads[0], compile_use) if open_reads else None

    def rule_out(env):
        nonlocal steps
        steps += 1
        if steps > budget:
            return None
        memo: dict = {}
        known = {c: cond.value(c, env, memo) for c in frames}
        if any(known[c] == 0 for c in use):
            return None
        open_decls = [d for d in decls if all(known[c] != 0 for c in d)]
        if any(all(known[c] == 1 for c in d) for d in open_decls):
            return None
        if not open_decls:
            return compile_use(env)
        # The use's own tests first (they usually settle the declarations), then the most-read parameter.
        votes: dict[str, int] = {}
        for c in [c for c in use if known[c] is None] + [c for d in open_decls for c in d if known[c] is None]:
            for n in cond.open_reads(c, env, memo) - use_only:
                votes[n] = votes.get(n, 0) + (1000 if c in use else 1)
        if not votes:
            return None
        return branch(env, min(votes, key=lambda n: (-votes[n], n)), rule_out)

    got = rule_out({})
    return got, steps <= budget


def checker(params: list[Param], constraints: list[Constraint]) -> Callable[[dict, str], bool]:
    """allowed(row, changed): no constraint on `changed` is broken by `row`, whatever its params not in the row
    become."""
    values = {p.name: p.values for p in params}
    watch: dict[str, list[Constraint]] = {}
    for c in constraints:
        for n in c.params:
            watch.setdefault(n, []).append(c)

    def allowed(row: dict, changed: str) -> bool:
        for c in watch.get(changed, ()):
            free = [n for n in c.params if n not in row]
            if not any(c.ok({**row, **dict(zip(free, vs))}) for vs in itertools.product(*(values[n] for n in free))):
                return False
        return True
    return allowed


def covering(params: list[Param], groups: list[tuple[str, ...]], constraints: list[Constraint],
             extra: list[dict] = ()) -> tuple[list[dict[str, int]], list[tuple], list[tuple]]:
    """Rows (param -> value) holding every value of every param, every pair of values of every two params, and every
    triple of values of every three params in one group; and the tuples left out, those a constraint forbids and those
    the greedy fill could not complete into an allowed row. `extra`: assignments of any params, each to be held by
    a row too. Greedy and deterministic: a row starts from the first uncovered tuple (longest first) and fills the
    other params, largest domains first, with the value completing the most uncovered tuples (ties: the smaller)."""
    names = [p.name for p in params]
    index = {n: i for i, n in enumerate(names)}
    values = {p.name: p.values for p in params}
    allowed = checker(params, constraints)

    combos = {(n,) for n in names} | set(itertools.combinations(names, 2))
    for g in groups:
        combos |= set(itertools.combinations(sorted(g, key=index.get), 3))
    wanted = {(c, v) for c in combos for v in itertools.product(*(values[n] for n in c))}
    for env in extra:  # given assignments of any number of params, each to be held by a row
        combo = tuple(sorted(env, key=index.get))
        combos.add(combo)
        wanted.add((combo, tuple(env[n] for n in combo)))
    seeds = sorted(wanted, key=lambda t: (-len(t[0]), [index[n] for n in t[0]], t[1]))
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
        cover([(c, vals) for c in combos if open_vals[c]
               for vals in [tuple(row[m] for m in c)] if vals in open_vals[c]])
        rows.append({n: row[n] for n in names})
    return rows, ruled_out, unplaced


@dataclasses.dataclass
class Stats:
    rows: int
    ruled_out: list[tuple]
    unplaced: list[tuple]
    duplicates: int
    lines: int = 0  # lines of tfx.glsl this stage can compile (no #if around them is false for every variant)
    held: int = 0  # of those, the ones some variant compiles
    unbuildable: list[int] = dataclasses.field(default_factory=list)  # first lines of blocks no game build compiles
    gave_up: list[int] = dataclasses.field(default_factory=list)  # first lines of blocks whose search gave up


def complete(env: dict, params: list[Param], allowed: Callable[[dict, str], bool]) -> dict | None:
    """env with every other param set to its first allowed value, or None if the constraints leave none."""
    row = dict(env)
    for p in params:
        if p.name in row:
            continue
        for v in p.values:
            row[p.name] = v
            if allowed(row, p.name):
                break
        else:
            return None
    return {p.name: row[p.name] for p in params}


def hold_every_line(stage: Stage, rows: list[dict]) -> tuple[list[dict], int, int, list[int], list[int], list]:
    """Rows to add so every line of tfx.glsl the stage can compile is compiled by some row: for each block (a
    distinct stack of #if conditions) no row compiles, the search finds parameter values compiling it, or proves no
    game build does. Returns the rows, the lines that can be compiled, those held, the first line of each unbuildable
    block, of each block whose search gave up, and the values the fill could not complete."""
    cond = stage.cond
    live = [(n, stack) for n, stack, line in cond.lines if line.strip() and cond.stack_value(stack, {}) != 0]
    first: dict[tuple, int] = {}
    for n, stack in live:
        first.setdefault(stack, n)
    open_stacks = set(first)
    for row in rows:
        memo: dict = {}
        open_stacks = {st for st in open_stacks if cond.stack_value(st, row, memo) != 1}
    allowed = checker(stage.params, stage.constraints)
    added, unbuildable, gave_up, unplaced = [], [], [], []
    for stack in sorted(open_stacks, key=first.get):
        if any(cond.stack_value(stack, row) == 1 for row in added):
            continue
        env, finished = find_missing_declaration(cond, stack, [], stage.params, stage.constraints, allowed)
        if env is None:
            (unbuildable if finished else gave_up).append(first[stack])
            continue
        row = complete(env, stage.params, allowed)
        if row is None:
            unplaced.append(((tuple(env)), tuple(env.values())))
        else:
            added.append(row)
    held_stacks = set(first) - set(open_stacks)
    held_stacks |= {st for st in open_stacks if any(cond.stack_value(st, row) == 1 for row in added)}
    held = sum(stack in held_stacks for _, stack in live)
    return added, len(live), held, unbuildable, gave_up, unplaced


def variants(src: Sources) -> tuple[list[Variant], dict[str, Stats]]:
    """Every stage's variants (one per distinct source), and per stage how they were made."""
    out, stats = [], {}
    for stage in src.stages:
        rows, ruled_out, unplaced = covering(stage.params, stage.groups, stage.constraints,
                                             [env for _, _, env in stage.missing])
        added, lines, held, unbuildable, gave_up, unfilled = hold_every_line(stage, rows)
        rows += added
        unplaced += unfilled
        seen, dupes = set(), 0
        for row in rows:
            text, hdr = assemble(src, stage, row)
            if text in seen:
                dupes += 1
                continue
            seen.add(text)
            out.append(Variant(stage.recipe.stage, row, text, hdr))
        stats[stage.recipe.stage] = Stats(len(rows), ruled_out, unplaced, dupes, lines, held, unbuildable,
                                          gave_up)
    return out, stats


# ---------------------------------------------------------------------------------------------------------------------
# Compiling.


def compile_one(compiler: list[str], v: Variant, debug: bool = False) -> tuple[Variant, int, str]:
    """debug: VKShaderCache's GSConfig.UseDebugDevice path (no optimisation, debug info), `tfxcc <stage> debug`."""
    r = subprocess.run(compiler + [v.stage] + (["debug"] if debug else []), input=v.source.encode(),
                       capture_output=True)
    return v, r.returncode, (r.stderr + r.stdout).decode(errors="replace")


PROBES = [  # (stage, source, exit code the compiler must give)
    ("vert", "#version 460 core\nvoid main() { gl_Position = vec4(0.0); }\n", 0),
    ("frag", "#version 460 core\nlayout(location = 0) out vec4 o;\nvoid main() { o = vec4(1.0); }\n", 0),
    ("frag", "#version 460 core\n#error check_tfx_variants probe: this source must not compile\nvoid main() {}\n", 1),
]


def probe(compiler: list[str], modes: tuple[bool, ...] = (False,)) -> str | None:
    """None if the compiler passes a good source and fails a bad one, in each mode (debug or not); else what went
    wrong. A compiler that always exits 0 would otherwise pass every variant."""
    for debug in modes:
        for stage, source, want in PROBES:
            v = Variant(stage, {}, source, 0)
            try:
                _, rc, msg = compile_one(compiler, v, debug)
            except OSError as e:
                return f"cannot run {shlex.join(compiler)}: {e}"
            if rc != want:
                cmd = shlex.join(compiler + [stage] + (["debug"] if debug else []))
                return f"compiler probe: {cmd} exited {rc} for a source that should give {want}:\n{source}{msg}"
    return None


def tfx_lines(message: str, header_lines: int) -> str:
    """`source:123:` (a line of the assembled source) also as the tfx.glsl line."""
    return re.sub(r"\bsource:(\d+):", lambda m: f"source:{m.group(1)} (tfx.glsl:{int(m.group(1)) - header_lines}):",
                  message)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compiler", default="tfxcc",
                    help="tfxcc, or any command taking vert|frag [debug], the source on stdin, exit 0 = compiled")
    ap.add_argument("--debug", action="store_true",
                    help="also compile every variant as with UseDebugDevice (no optimisation, debug info)")
    ap.add_argument("--tfx", type=Path, help=f"the shader to check instead of {TFX} (a scratch copy)")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--jobs", type=int, default=0, help="compiles at once (0: one per CPU)")
    ap.add_argument("--list", action="store_true", help="print the parameters and variants, compile nothing")
    ap.add_argument("--dump", type=Path, help="write every variant's source here")
    args = ap.parse_args(argv)
    args.jobs = args.jobs if args.jobs > 0 else os.cpu_count() or 4

    start = time.monotonic()
    compiler = shlex.split(args.compiler)
    if not args.list:  # before the minutes of work, the seconds that make them mean something
        problem = probe(compiler, (False, True) if args.debug else (False,))
        if problem:
            print(f"check_tfx_variants: {problem}", file=sys.stderr)
            return 2
    try:
        src = load(args.root, args.tfx)
        for c in CONSTRAINTS:
            cite(c, args.root)
    except SourceError as e:
        print(f"check_tfx_variants: {e}", file=sys.stderr)
        return 2
    vs, stats = variants(src)

    for w in src.warnings:
        print(f"WARNING {w}")
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

    failed = []
    jobs = [(v, False) for v in vs] + ([(v, True) for v in vs] if args.debug else [])
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for (v, debug), (_, rc, msg) in zip(jobs, pool.map(lambda j: compile_one(compiler, *j), jobs)):
            if rc != 0:
                failed.append((v, debug, msg))
    for v, debug, msg in failed:
        print(f"FAIL{' (debug)' if debug else ''} {v.describe()}")
        for line in tfx_lines(msg, v.header_lines).strip().splitlines()[:12]:
            print(f"    {line}")
    unplaced = [(name, t) for name, st in stats.items() for t in st.unplaced]
    for name, (combo, vals) in unplaced:
        print(f"UNPLACED {name}: no variant holds {' '.join(f'{n}={v}' for n, v in zip(combo, vals))}; "
              f"a constraint missing from {Path(__file__).name}, or the fill needs to backtrack")
    gave_up = [f"{st.recipe.stage}: could not tell whether {name} at tfx.glsl:{line} can be compiled without its "
               f"declaration" for st in src.stages for name, line in st.gave_up]
    gave_up += [f"{name}: could not tell whether any game build compiles the block at tfx.glsl:{line}"
                for name, st in stats.items() for line in st.gave_up]
    for text in gave_up:
        print(f"GAVE UP {text}; raise the search budget in {Path(__file__).name} or simplify the conditions")
    per_stage = ", ".join(f"{name} {sum(v.stage == name for v in vs)}" for name in stats)
    print(f"tfx variants: {len(vs)} ({per_stage}){', each also as debug' if args.debug else ''}, "
          f"{len(failed)} failed; every pair of "
          f"{' / '.join(f'{len(st.params)} {st.recipe.stage}' for st in src.stages)} parameters and every triple in "
          f"{sum(len(st.groups) for st in src.stages)} #if groups, "
          f"{sum(len(st.missing) for st in src.stages)} uses without their declaration held, "
          f"lines held: {', '.join(f'{name} {st.held}/{st.lines}' for name, st in stats.items())}"
          f"{''.join(f' ({name}: {len(st.unbuildable)} blocks no game build compiles, from tfx.glsl:'
                     f'{st.unbuildable[0]})' for name, st in stats.items() if st.unbuildable)}, "
          f"{sum(len(st.ruled_out) for st in stats.values())} combinations ruled out by {len(CONSTRAINTS)} constraints")
    # Last, and the only line that differs between runs of the same sources (the Nix check keeps it out of $out).
    print(f"time: {args.jobs} jobs, {time.monotonic() - start:.1f}s")
    return 1 if failed or unplaced or gave_up else 0


if __name__ == "__main__":
    sys.exit(main())
