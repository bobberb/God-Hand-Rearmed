"""Gate tests of tools/linux/check_shared_state.py: what it must find, what it must leave alone, and the repo's own
headers (which must be clean: this is the gate in CI, through the Nix check's tools/linux test run).

    python -I -m unittest discover -s tools/linux -p "test_check_shared_state.py" -v
"""
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_shared_state  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def findings(header_text):
    """The checker's findings for one header placed in the runtime's include folder of a scratch tree."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "runtime/ps2xRuntime/include/sample.h"
        path.parent.mkdir(parents=True)
        path.write_text(textwrap.dedent(header_text))
        return [(number, kind) for number, _, kind in check_shared_state.scan(path, root)]


class Finds(unittest.TestCase):
    def test_inline_variable(self):
        self.assertEqual(findings("inline int counter = 0;\n"), [(1, "inline variable")])

    def test_inline_variable_without_initializer(self):
        # vu0NormShadow's shape: no '=' at all.
        self.assertEqual(findings("struct S { int x; };\ninline S shadow;\n"), [(2, "inline variable")])

    def test_inline_thread_local(self):
        self.assertEqual(findings("inline thread_local int *sink = nullptr;\n"), [(1, "inline variable")])

    def test_inline_variable_in_namespace(self):
        self.assertEqual(findings("namespace a\n{\n    inline int b = 0;\n}\n"), [(3, "inline variable")])

    def test_static_local_of_inline_function(self):
        text = """\
            inline int &value()
            {
                static int v = 0;
                return v;
            }
            """
        self.assertEqual(findings(text), [(3, "static local of a header function")])

    def test_static_local_brace_initialized(self):
        text = """\
            inline std::atomic<int *> &ptr()
            {
                static std::atomic<int *> p{nullptr};
                return p;
            }
            """
        self.assertEqual(findings(text), [(3, "static local of a header function")])

    def test_static_local_in_member_function(self):
        text = """\
            struct Log
            {
                static std::mutex &m()
                {
                    static std::mutex mutex;
                    return mutex;
                }
            };
            """
        self.assertEqual(findings(text), [(5, "static local of a header function")])

    def test_static_data_member_with_initializer(self):
        self.assertEqual(findings("struct S\n{\n    static inline int count = 0;\n};\n"), [(3, "inline variable")])

    def test_extern_variable_without_api(self):
        self.assertEqual(findings("extern int g_flags;\n"), [(1, "extern variable without PS2X_RUNTIME_API")])

    def test_static_in_shared_branch_still_counts(self):
        # The #else of `#if !PS2X_SHARED_STATE` is the split build's form: a static there is a finding.
        text = """\
            #if !PS2X_SHARED_STATE
            inline int a = 0;
            #else
            inline int b = 0;
            #endif
            """
        self.assertEqual(findings(text), [(4, "inline variable")])


class FindsTheReviewersForms(unittest.TestCase):
    """The forms a review (Nov 27) showed the first version missing: each one must now be found."""

    CASES = {
        "one-line accessor": 'inline int &counter() { static int c = 0; return c; }',
        "thread_local without static": 'inline void f()\n{\n    thread_local int buf = 0;\n    (void)buf;\n}',
        "alignas": 'inline void f()\n{\n    alignas(16) static float buf[4];\n    (void)buf;\n}',
        "attribute": 'inline void f()\n{\n    [[maybe_unused]] static int n = 0;\n}',
        "declaration over two lines": 'inline std::atomic<int> &f()\n{\n    static std::atomic<int>\n        n{0};\n    return n;\n}',
        "template static member": 'template <class T> struct S { static int n; };\ntemplate <class T> int S<T>::n = 0;',
        'extern "C" variable': 'extern "C" int g_cdata;',
        "#ifndef's other branch": '#ifndef PS2X_SHARED_STATE\n#else\ninline int g_x = 0;\n#endif',
        "one-line class": 'struct Q { static int k; };',
        "static member in an override": 'struct D : B\n{\n    int g() override\n    {\n        static int x = 0;\n        return x;\n    }\n};',
        "lambda": 'inline auto f = [] { static int c = 0; return ++c; };',
        "extern over two lines": 'extern std::atomic<int>\n    g_multi;',
        "class thread_local member": 'struct T\n{\n    static thread_local int t;\n};',
        "template function's static": 'template <class T>\nT &inst()\n{\n    static T t;\n    return t;\n}',
        "two variables": 'inline int g_a = 0, g_b = 0;',
        "one-line namespace": 'namespace n { inline int g_v = 0; }',
    }

    def test_each(self):
        for name, text in self.CASES.items():
            with self.subTest(name):
                self.assertTrue(findings(text + "\n"), f"not found: {name}")

    def test_normal_build_branch_spelled_with_defined(self):
        self.assertEqual(findings('#if !defined(PS2X_SHARED_STATE) || !PS2X_SHARED_STATE\ninline int g_ok = 0;\n#endif\n'),
                         [])


class LeavesAlone(unittest.TestCase):
    def test_constexpr(self):
        self.assertEqual(findings("inline constexpr int kSize = 4;\nstatic constexpr int kTable[2] = {1, 2};\n"), [])

    def test_inline_function(self):
        self.assertEqual(findings("inline int twice(int x)\n{\n    return 2 * x;\n}\n"), [])

    def test_function_declaration(self):
        self.assertEqual(findings("int helper(int x);\nextern \"C\" void hook();\nextern int f(int);\n"), [])

    def test_marked(self):
        text = """\
            inline bool flag()
            {
                static const bool enabled = true; // shared-state: an environment flag, read once per module
                return enabled;
            }
            """
        self.assertEqual(findings(text), [])

    def test_normal_build_branch(self):
        text = """\
            #if !PS2X_SHARED_STATE
            inline int counter = 0;
            inline int &value()
            {
                static int v = 0;
                return v;
            }
            #endif
            """
        self.assertEqual(findings(text), [])

    def test_annotated_extern(self):
        self.assertEqual(findings("extern PS2X_RUNTIME_API std::atomic<int> g_flags;\n"), [])

    def test_namespace_scope_static(self):
        # Internal linkage: every source file has its own copy on every platform; not a split-build hazard.
        self.assertEqual(findings("static const int columnTable[4] = {0, 1, 2, 3};\n"), [])

    def test_comments(self):
        text = """\
            // inline int counter = 0;
            /* static int v = 0;
               inline int b = 1; */
            """
        self.assertEqual(findings(text), [])


class RepoHeaders(unittest.TestCase):
    def test_repo_is_clean(self):
        found = []
        for path in check_shared_state.headers(REPO):
            found += [f"{path.relative_to(REPO)}:{n}: {kind}" for n, _, kind in check_shared_state.scan(path, REPO)]
        self.assertEqual(found, [], "per-module definitions in the game code's headers (see check_shared_state.py)")

    def test_scans_the_known_headers(self):
        names = {p.relative_to(REPO).as_posix() for p in check_shared_state.headers(REPO)}
        for expected in ("runtime/ps2xRuntime/include/ps2_log.h", "runtime/ps2xRuntime/include/runtime/ps2_memory.h",
                         "runtime/ps2xRuntime/src/lib/vu/ps2_vu1_impl.inl", "src/kamite/module_registry.h"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
