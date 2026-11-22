"""Gate tests of check_tfx_variants.py's pure parts: reading GSDeviceVK.cpp / GSDevice.h / tfx.glsl, the value
classes, the covering array, the source assembly, the report. No shader compiler: a shell one-liner stands in.

    python -I -m unittest discover -s tools/linux -p "test_*.py" -v
"""
import contextlib
import dataclasses
import io
import itertools
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_tfx_variants as ctv  # noqa: E402

ROOT = ctv.ROOT
SH = shutil.which("sh")


def satisfied(c: ctv.Constraint, row: dict) -> bool:
    return c.ok({n: row[n] for n in c.params})


class RealSources(unittest.TestCase):
    """The repository's own GSDeviceVK.cpp, GSDevice.h and tfx.glsl."""

    @classmethod
    def setUpClass(cls):
        cls.src = ctv.load()
        cls.vs, cls.stats = ctv.variants(cls.src)
        cls.stage = {s.recipe.stage: s for s in cls.src.stages}

    def test_two_stages_with_the_kamite_macros(self):
        self.assertEqual([s.recipe.stage for s in self.src.stages], ["vert", "frag"])
        vert = [m.name for m in self.stage["vert"].recipe.macros]
        frag = [m.name for m in self.stage["frag"].recipe.macros]
        self.assertIn("VS_PERSP_COLOR", vert)
        self.assertIn("PS_SEE_THROUGH", frag)
        self.assertIn("PS_PERSP_COLOR", frag)
        provoking = next(m for m in self.stage["vert"].recipe.macros if m.name == "VS_PROVOKING_VERTEX_LAST")
        self.assertEqual(provoking.feature, "features.provoking_vertex_last")

    def test_macros_in_the_order_gsdevicevk_adds_them(self):
        cpp = (ROOT / ctv.DEVICE_CPP).read_text()
        for stage, function in (("vert", "GetTFXVertexShader"), ("frag", "GetTFXFragmentShader")):
            body = cpp[cpp.index(f"GSDeviceVK::{function}("):]
            body = body[:body.index("ss << m_tfx_source;")]
            want = re.findall(r'AddMacro\(ss, "(\w+)"', body)
            self.assertEqual([m.name for m in self.stage[stage].recipe.macros], want)

    def test_value_domains_from_the_selector_structs(self):
        values = {p.name: p.values for p in self.stage["frag"].params}
        self.assertEqual(values["PS_ATST"], (0, 1, 2, 3, 4))  # the PS_ATST enum
        self.assertEqual(values["PS_AFAIL"], (0, 1, 2, 3, 4, 5))
        self.assertEqual(values["PS_SEE_THROUGH"], (0, 1))
        self.assertEqual(values["PS_ZTST"], (0, 2, 3))  # tfx.glsl tests GEQUAL and GREATER only: ALWAYS = NEVER
        self.assertEqual(values["features.rov"], (0, 1))
        self.assertEqual(dict((p.name, p.values) for p in self.stage["vert"].params)["VS_EXPAND"], tuple(range(6)))

    def test_one_parameter_constraints_narrow_the_domain(self):
        values = {p.name: p.values for p in self.stage["frag"].params}
        # sw_aniso is 5 bits, but the game builds 0..16 (Bridge.cpp clamps MaxAnisotropy): 16, not 31, is the
        # largest compiled, the value an array sized by it sees.
        self.assertEqual(values["PS_ANISOTROPIC_FILTERING"], (0, 1, 2, 16))
        self.assertEqual(values["PS_DATE"], (0, 1, 2, 3, 5, 6))
        self.assertFalse(any(v.values.get("PS_ANISOTROPIC_FILTERING", 0) > 16 for v in self.vs))

    def test_assembled_source_is_header_stage_macros_shader(self):
        stage = self.stage["frag"]
        row = {p.name: p.values[0] for p in stage.params}
        row.update({"features.texture_barrier": 0, "features.UseFeedbackLoopLayout": 0, "features.rov": 0})
        text, hdr = ctv.assemble(self.src, stage, row)
        self.assertTrue(text.startswith("#version 460 core\n#extension GL_EXT_samplerless_texture_functions : require\n"
                                        "#define DISABLE_TEXTURE_BARRIER 1\n#define FRAGMENT_SHADER 1\n#define PS_FST 0\n"))
        self.assertTrue(text.endswith(self.src.tfx))
        self.assertEqual(text[:len(text) - len(self.src.tfx)].count("\n"), hdr)
        self.assertEqual(hdr, 4 + len(stage.recipe.macros))

        row.update({"features.texture_barrier": 1, "features.UseFeedbackLoopLayout": 1, "features.rov": 1})
        text, _ = ctv.assemble(self.src, stage, row)
        self.assertNotIn("DISABLE_TEXTURE_BARRIER", text[:200])
        self.assertIn("#define HAS_FEEDBACK_LOOP_LAYOUT 1\n#extension GL_ARB_fragment_shader_interlock : require\n"
                      "#extension GL_ARB_shader_image_load_store : require\n#define FRAGMENT_SHADER 1\n", text)

        vert = self.stage["vert"]
        row = {p.name: 0 for p in vert.params} | {"features.provoking_vertex_last": 1, "VS_EXPAND": 3}
        text, _ = ctv.assemble(self.src, vert, row)
        self.assertIn("#define VERTEX_SHADER 1\n", text)
        self.assertIn("#define VS_EXPAND 3\n#define VS_PROVOKING_VERTEX_LAST 1\n#define VS_PERSP_COLOR 0\n", text)

    def test_every_variant_is_one_the_game_can_build(self):
        for v in self.vs:
            for c in self.stage[v.stage].constraints:
                self.assertTrue(satisfied(c, v.values), f"{v.describe()} breaks {c.why}")

    def test_coverage_every_pair_and_group_triple(self):
        for name, stage in self.stage.items():
            rows = [v.values for v in self.vs if v.stage == name]
            ruled_out = set(self.stats[name].ruled_out)
            self.assertEqual(self.stats[name].unplaced, [])
            names = [p.name for p in stage.params]
            index = {n: i for i, n in enumerate(names)}
            combos = list(itertools.combinations(names, 2))
            for g in stage.groups:
                combos += itertools.combinations(sorted(g, key=index.get), 3)
            values = {p.name: p.values for p in stage.params}
            seen = {(c, tuple(r[n] for n in c)) for c in set(combos) for r in rows}
            for c in set(combos):
                for vals in itertools.product(*(values[n] for n in c)):
                    if (c, vals) not in seen:
                        self.assertIn((c, vals), ruled_out, f"{name}: no variant holds {dict(zip(c, vals))}")

    def test_ruled_out_means_a_constraint_forbids_it(self):
        for name, stage in self.stage.items():
            values = {p.name: p.values for p in stage.params}
            for combo, vals in self.stats[name].ruled_out:
                fixed = dict(zip(combo, vals))

                def forbids(c):
                    free = [n for n in c.params if n not in fixed]
                    return all(not c.ok({**fixed, **dict(zip(free, vs))})
                               for vs in itertools.product(*(values[n] for n in free)))
                self.assertTrue(any(forbids(c) for c in stage.constraints if set(c.params) & set(combo)), fixed)

    def test_primid_pass_variants_as_gsdevicevk_builds_them(self):
        primid = [v.values for v in self.vs if v.stage == "frag" and v.values["PS_DATE"] in (1, 2)]
        self.assertTrue(primid)
        for v in primid:
            self.assertEqual((v["PS_NO_COLOR"], v["PS_NO_COLOR1"]), (0, 1))
            self.assertEqual([v[f"PS_BLEND_{x}"] for x in "ABCD"], [0, 0, 0, 0])

    def test_both_values_of_see_through_are_compiled(self):
        seen = {v.values["PS_SEE_THROUGH"] for v in self.vs if v.stage == "frag"}
        self.assertEqual(seen, {0, 1})

    def test_groups_hold_the_nested_feedback_loop_block(self):
        # The subpassInput declarations: under #else of DISABLE_TEXTURE_BARRIER || HAS_FEEDBACK_LOOP_LAYOUT, inside
        # PS_FEEDBACK_LOOP_IS_NEEDED_RT || ..._DEPTH, reading PS_ROV_COLOR.
        groups = [set(g) for g in self.stage["frag"].groups]
        want = {"features.texture_barrier", "features.UseFeedbackLoopLayout", "PS_ROV_COLOR", "PS_FBMASK", "PS_ZTST"}
        self.assertTrue(any(want <= g for g in groups))

    def test_variant_sources_are_distinct(self):
        self.assertEqual(len({v.source for v in self.vs}), len(self.vs))

    def test_constraint_citations_point_at_their_lines(self):
        for c in ctv.CONSTRAINTS:
            self.assertTrue(c.cites, c.why)
            places = ctv.cite(c).split(", ")
            self.assertEqual(len(places), sum(n for _, _, n in c.cites))
            for place in places:
                path, line = place.rsplit(":", 1)
                text = (ROOT / path).read_text().splitlines()[int(line) - 1]
                self.assertTrue(any(p == path and re.search(a, text) for p, a, _ in c.cites), place)

    def test_citations_named_in_the_critique(self):
        cited = {c.params: ctv.cite(c) for c in ctv.CONSTRAINTS}
        self.assertIn(f"{ctv.DEVICE_CPP}:5148", cited[("PS_ROV_COLOR", "PS_NO_COLOR1")])
        for line in (f"{ctv.RENDERER_HW}:6771", f"{ctv.RENDERER_HW}:6779", f"{ctv.RENDERER_HW}:8404",
                     f"{ctv.DEVICE_CPP}:6154"):
            self.assertIn(line, cited[("PS_DATE",)])

    def test_a_stale_citation_stops(self):
        c = dataclasses.replace(ctv.CONSTRAINTS[0], cites=((ctv.DEVICE_CPP, r"pipe\.ps\.no_color", 1),))
        with self.assertRaisesRegex(ctv.SourceError, r"matches [02-9]\d* lines of .*, want 1"):
            ctv.cite(c)


class ValueClasses(unittest.TestCase):
    def uses(self, glsl, *macros):
        return ctv.macro_uses(glsl, set(macros))

    def test_comparisons_split_the_domain(self):
        u = self.uses("#if M == 1\n#elif M >= 4\n#endif\n", "M")["M"]
        self.assertEqual(u.atoms, {("==", 1), (">=", 4)})
        self.assertEqual(ctv.value_classes(list(range(8)), u), [0, 1, 4])

    def test_named_constants_bitwise_and_flipped_operands(self):
        u = self.uses("#define K 2\n#if (M & K) && 3 < M\n#endif\n", "M")["M"]
        self.assertEqual(u.atoms, {("&", 2), (">", 3)})
        self.assertEqual(ctv.value_classes(list(range(8)), u), [0, 2, 4, 6])

    def test_bare_tests_and_values_in_code(self):
        u = self.uses("#if M\n#endif\nfloat x = min(y, M);\n", "M")["M"]
        self.assertEqual(ctv.value_classes(list(range(32)), u), [0, 1, 31])

    def test_compared_with_another_macro_keeps_every_value(self):
        u = self.uses("#if A == B\n#endif\n", "A", "B")
        self.assertTrue(u["A"].exact and u["B"].exact)
        self.assertEqual(ctv.value_classes([0, 1, 2, 3], u["A"]), [0, 1, 2, 3])

    def test_definitions_defined_and_comments_are_not_uses(self):
        u = self.uses("#ifndef M\n#define M 0\n#endif\n#if defined(M)\n#endif\n// M == 3\n/* M == 4 */\n", "M")["M"]
        self.assertEqual(u.atoms, set())
        self.assertEqual(ctv.value_classes([0, 1, 2], u), [0])

    def test_hex_suffix_and_parentheses_are_read(self):
        u = self.uses("#define K (0x4u)\n#if M == 0x10\n#elif (M) >= 2u\n#elif M < (K)\n#endif\n", "M")["M"]
        self.assertEqual(u.atoms, {("==", 16), (">=", 2), ("<", 4)})
        self.assertIn(16, ctv.value_classes(list(range(32)), u))

    def test_unreadable_comparisons_stop(self):
        for glsl in ("#if M == 8 * 2\n#endif\n",                          # an expression
                     "#if M + 1 == 3\n#endif\n",
                     "#define IS(x) (M == x)\n",                         # a function-like macro's parameter
                     "#define K (8 << 1)\n#if M == K\n#endif\n",         # a constant that is an expression
                     "#define ALIAS M\n",                                # an alias, compared who knows where
                     "#if !M == 1\n#endif\n"):
            with self.subTest(glsl=glsl), self.assertRaisesRegex(ctv.SourceError, r"tfx\.glsl:\d+: M is read as"):
                self.uses(glsl, "M")

    def test_truth_tests_in_every_position(self):
        u = self.uses("#if M\n#elif !M && (M) || (M || X)\n#endif\n#define D (M && X)\n", "M")["M"]
        self.assertEqual(u.atoms, {("!=", 0)})

    def test_critic_repro_hex_value_is_compiled(self):
        tfx = (ROOT / ctv.TFX).read_text().replace(
            "#if PS_ZTST == ZTST_GEQUAL", "#if PS_ANISOTROPIC_FILTERING == 0x10\nkamite_bad = 1;\n#endif\n"
            "#if PS_ZTST == ZTST_GEQUAL", 1)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tfx.glsl").write_text(tfx)
            src = ctv.load(tfx_path=Path(d) / "tfx.glsl")
        af = next(p for p in src.stages[1].params if p.name == "PS_ANISOTROPIC_FILTERING")
        self.assertIn(16, af.values)

    def test_prefix_names_are_other_macros(self):
        u = self.uses("#if PS_ATST_NONE == 1\n#endif\n", "PS_ATST")["PS_ATST"]
        self.assertEqual(u.atoms, set())


class ReadingTheCpp(unittest.TestCase):
    HEADER = """
static void AddShaderHeader(std::stringstream& ss)
{
	const GSDeviceVK* dev = GSDeviceVK::GetInstance();
	ss << "#version 460 core\\n";
	if (!features.a)
		ss << "#define NO_A 1\\n";
	if (features.a && dev->B())
	{
		ss << "#extension X : require\\n";
		ss << "#define AB 1\\n";
	}
}
"""

    def test_header_lines_and_their_guards(self):
        lines, features = ctv.parse_header(self.HEADER)
        self.assertEqual(features, {"features.a", "features.B"})
        self.assertEqual([h.text for h in lines], ["#version 460 core\n", "#define NO_A 1\n", "#extension X : require\n",
                                                   "#define AB 1\n"])
        F = {"features.a": True, "features.B": False}
        self.assertEqual([h.guard is None or eval(h.guard, {}, {"F": F}) for h in lines], [True, False, False, False])

    def test_unknown_header_statement_stops(self):
        with self.assertRaisesRegex(ctv.SourceError, "not understood"):
            ctv.parse_header(self.HEADER.replace('ss << "#version', 'ss << version << "#version'))
        with self.assertRaisesRegex(ctv.SourceError, "other than device features"):
            ctv.parse_header(self.HEADER.replace("!features.a", "!features.a || GSConfig.X"))

    def test_add_macro_values(self):
        self.assertEqual(ctv.parse_macro_arg("sel.tme", "x"), ("tme", None))
        self.assertEqual(ctv.parse_macro_arg("static_cast<u32>(sel.atst)", "x"), ("atst", None))
        self.assertEqual(ctv.parse_macro_arg("static_cast<int>(m_features.pvl)", "x"), (None, "features.pvl"))
        with self.assertRaises(ctv.SourceError):
            ctv.parse_macro_arg("sel.tme && GSConfig.X", "x")

    ADD_MACRO = 'ss << "#define " << name << " " << value << "\\n";'

    def test_macro_lines_follow_the_add_macro_helper(self):
        cpp = (ROOT / ctv.DEVICE_CPP).read_text()
        self.assertIn(self.ADD_MACRO, cpp)
        self.assertEqual(ctv.parse_add_macro(cpp), ["#define ", "{name}", " ", "{value}", "\n"])
        src = ctv.load()
        frag = src.stages[1]
        row = {p.name: p.values[0] for p in frag.params}
        for helper, want in (('ss << "#define " << name << " " << value << "u\\n";', "#define PS_FST 0u\n"),
                             ('ss << "#define " << name << " (" << value << ")\\n";', "#define PS_FST (0)\n")):
            fmt = "".join(ctv.parse_add_macro(cpp.replace(self.ADD_MACRO, helper)))
            text, _ = ctv.assemble(dataclasses.replace(src, macro_format=fmt), frag, row)
            self.assertIn(want, text)

    def test_add_macro_helper_of_another_shape_stops(self):
        cpp = (ROOT / ctv.DEVICE_CPP).read_text()
        for bent in (cpp.replace(self.ADD_MACRO, 'ss << "#define " << name << " " << std::hex << value << "\\n";'),
                     cpp.replace(self.ADD_MACRO, 'ss << "#define " << name << "\\n";'),
                     cpp.replace(self.ADD_MACRO, self.ADD_MACRO + "\n\tss << value;"),
                     cpp.replace("const char* name, int value)", "const char* name, bool value)", 1)):
            with self.assertRaises(ctv.SourceError):
                ctv.parse_add_macro(bent)

    def test_stage_macro_lines(self):
        cpp = ctv.strip_c_comments((ROOT / ctv.DEVICE_CPP).read_text())
        self.assertEqual(ctv.parse_stage_macros(cpp), {"vs": "#define VERTEX_SHADER 1\n",
                                                       "gs": "#define GEOMETRY_SHADER 1\n",
                                                       "fs": "#define FRAGMENT_SHADER 1\n"})
        first = '\tif (vs)\n\t\tss << "#define VERTEX_SHADER 1\\n";'
        self.assertIn(first, cpp)
        for bent in (cpp.replace(first, '\tss << "#define ANY_STAGE 1\\n";\n' + first),   # unconditional
                     cpp.replace(first, first + '\n\t\tss << "#define AGAIN 1\\n";'),      # a second line
                     cpp.replace("else if (gs)", "if (gs)", 1)):                          # not one chain
            with self.assertRaisesRegex(ctv.SourceError, "AddShaderStageMacro"):
                ctv.parse_stage_macros(bent)

    def test_conditional_add_macro_stops(self):
        cpp = ctv.strip_c_comments((ROOT / ctv.DEVICE_CPP).read_text())
        self.assertEqual(len(ctv.parse_stages(cpp)), 2)
        bent = cpp.replace('\tAddMacro(ss, "PS_SEE_THROUGH"', '\tif (GSConfig.X)\n\t\tAddMacro(ss, "PS_SEE_THROUGH"', 1)
        with self.assertRaisesRegex(ctv.SourceError, r"GSDeviceVK\.cpp:\d+: GetTFXFragmentShader statement 'if"):
            ctv.parse_stages(bent)

    def test_enum_members_implicit_and_explicit(self):
        enums = ctv.parse_enums("enum class A : u32\n{\n\tX = 0, // c\n\tY,\n\tZ = 0x4,\n\tW\n};\n"
                                "enum class B : u8 { P = 1 << 1, Q };\n")
        self.assertEqual(enums["A"], [0, 1, 4, 5])
        self.assertIsInstance(enums["B"], ctv.SourceError)
        device_h = "using B = GSShader::B;\nstruct S { union { struct { u32 a : 1; B b : 2; }; }; };\n"
        with self.assertRaisesRegex(ctv.SourceError, "1 << 1"):
            ctv.parse_selector(device_h, "S", enums)

    def test_real_afail_gains_an_added_member(self):
        enums_h = (ROOT / ctv.ENUMS_H).read_text()
        bent = enums_h.replace("RGB_ONLY_SW_Z = 5, ///< RGB only with software Z discard",
                               "RGB_ONLY_SW_Z = 5, ///< RGB only with software Z discard\n\tRGB_ONLY_NEW,")
        self.assertNotEqual(bent, enums_h)
        self.assertEqual(ctv.parse_enums(bent)["PS_AFAIL"], [0, 1, 2, 3, 4, 5, 6])

    def test_selector_bitfields_and_enums(self):
        device_h = """
using E = GSShader::E;
struct S { union { struct { u32 a : 2; E e : 2; u8 b : 1; }; struct { u64 key; }; }; };
"""
        self.assertEqual(ctv.parse_selector(device_h, "S", {"E": [0, 1, 5]}), {"a": [0, 1, 2, 3], "e": [0, 1], "b": [0, 1]})


class Groups(unittest.TestCase):
    def test_nesting_elif_and_expanded_defines(self):
        header = [ctv.HeaderLine("#define NO_BARRIER 1\n", ' not F["features.barrier"] ')]
        names = {"A", "B", "C", "D", "features.barrier"}
        glsl = """#define AB (A || B)
#if AB
  #if C
  #endif
#elif D
#endif
#ifdef NO_BARRIER
  #if A && C
  #endif
#endif
#ifndef A
#endif
"""
        self.assertEqual(ctv.interaction_groups(glsl, header, names),
                         [("A", "B", "C"), ("A", "B", "D"), ("A", "C", "features.barrier")])

    def test_unbalanced(self):
        with self.assertRaises(ctv.SourceError):
            ctv.interaction_groups("#if A\n", [], {"A"})
        with self.assertRaises(ctv.SourceError):
            ctv.interaction_groups("#endif\n", [], {"A"})


class Covering(unittest.TestCase):
    PARAMS = [ctv.Param("A", (0, 1, 2)), ctv.Param("B", (0, 1)), ctv.Param("C", (0, 1)), ctv.Param("D", (0, 1, 2, 3))]
    CONS = [ctv.Constraint("frag", ("A", "B", "C"), lambda v: not (v["A"] == 2 and v["B"]) or v["C"], "x", ()),
            ctv.Constraint("frag", ("C", "D"), lambda v: not (v["C"] and v["D"] == 3), "y", ())]

    def test_pairs_triples_constraints(self):
        rows, ruled_out, unplaced = ctv.covering(self.PARAMS, [("A", "B", "D")], self.CONS)
        for r in rows:
            self.assertTrue(all(satisfied(c, r) for c in self.CONS), r)
        # C=1 with D=3: one constraint forbids it.
        self.assertEqual(ruled_out, [(("C", "D"), (1, 3))])
        # A=2, B=1 needs C=1, which cannot meet D=3: only the two constraints together say so, which the per-constraint
        # check cannot see, so the fill fails and main() reports it (UNPLACED, exit 1) instead of dropping it quietly.
        self.assertEqual(unplaced, [(("A", "B", "D"), (2, 1, 3))])
        values = {p.name: p.values for p in self.PARAMS}
        combos = [(n,) for n in "ABCD"] + list(itertools.combinations("ABCD", 2)) + [("A", "B", "D")]
        for c in combos:
            for vals in itertools.product(*(values[n] for n in c)):
                if (c, vals) not in ruled_out + unplaced:
                    self.assertTrue(any(all(r[n] == v for n, v in zip(c, vals)) for r in rows), (c, vals))
        self.assertGreaterEqual(len(rows), 3 * 2 * 4 - 1)  # the A x B x D triples, less the one left out

    def test_deterministic(self):
        self.assertEqual(ctv.covering(self.PARAMS, [("A", "B", "D")], self.CONS),
                         ctv.covering(self.PARAMS, [("A", "B", "D")], self.CONS))


OK_COMPILER = f"{SH} -c '! grep -q \"^#error\"' fake"  # compiles all but the probe's bad source


class Report(unittest.TestCase):
    def test_line_numbers_also_as_tfx_lines(self):
        self.assertEqual(ctv.tfx_lines("source:120: error: 'bad' : undeclared\n", 70),
                         "source:120 (tfx.glsl:50): error: 'bad' : undeclared\n")

    @unittest.skipUnless(SH, "needs sh")
    def test_failing_variants_named_and_exit_1(self):
        # A stand-in compiler that fails exactly the variants with the see-through oval on.
        fake = f"{SH} -c '! grep -q -e \"^#define PS_SEE_THROUGH 1$\" -e \"^#error\"' fake"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = ctv.main(["--compiler", fake])
        self.assertEqual(rc, 1)
        fails = [line for line in out.getvalue().splitlines() if line.startswith("FAIL ")]
        src = ctv.load()
        vs, _ = ctv.variants(src)
        want = [v for v in vs if v.values.get("PS_SEE_THROUGH")]
        self.assertEqual(len(fails), len(want))
        self.assertTrue(all(f.startswith("FAIL frag ") and "PS_SEE_THROUGH=1" in f for f in fails))
        self.assertRegex(out.getvalue().splitlines()[-2],
                         rf"^tfx variants: {len(vs)} \(vert \d+, frag \d+\), {len(want)} failed; ")

    @unittest.skipUnless(SH, "needs sh")
    def test_report_is_the_same_bytes_but_for_the_time_line(self):
        runs = []
        for jobs in ("1", "7"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ctv.main(["--compiler", OK_COMPILER, "--jobs", jobs])
            lines = out.getvalue().splitlines()
            self.assertRegex(lines[-1], r"^time: \d+ jobs, [\d.]+s$")
            self.assertFalse(any(line.startswith("time: ") for line in lines[:-1]))
            runs.append(lines[:-1])
        self.assertEqual(runs[0], runs[1])
        flake = (ROOT / "flake.nix").read_text()
        check = flake[flake.index('tfxVariants = pkgs.runCommand'):]
        check = check[:check.index("'';")]
        self.assertIn("grep -v '^time: ' report.txt > $out", check)
        self.assertNotIn("tee $out", check)

    @unittest.skipUnless(SH, "needs sh")
    def test_a_compiler_that_cannot_fail_or_cannot_pass_stops(self):
        for compiler in (f"{SH} -c 'cat > /dev/null' fake", f"{SH} -c 'cat > /dev/null; exit 1' fake",
                         "/nonexistent/tfxcc"):
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                self.assertEqual(ctv.main(["--compiler", compiler]), 2, compiler)
            self.assertRegex(err.getvalue(), "compiler probe|cannot run")

    @unittest.skipUnless(SH, "needs sh")
    def test_all_compiling_exit_0(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(ctv.main(["--compiler", OK_COMPILER]), 0)
        self.assertNotIn("FAIL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
