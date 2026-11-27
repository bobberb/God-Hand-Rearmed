"""Gate tests of tools/linux/check_shared_symbols.py, through a fake nm with canned output: what passes, what fails
(an unlisted symbol, a stale allow-list entry, an object without its inline functions, an entry without a reason).

    python -I -m unittest discover -s tools/linux -p "test_check_shared_symbols.py" -v
"""
import io
import stat
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_shared_symbols  # noqa: E402

FUNCTIONS = "".join(f"00000000000{i:05x} W inline_function_{i}\n" for i in range(150))


class Fixture:
    """A scratch folder holding a fake nm (prints `demangled` with -C, else `raw`) and an allow-list."""

    def __init__(self, demangled, allow, raw=FUNCTIONS):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "demangled.txt").write_text(demangled)
        (root / "raw.txt").write_text(raw)
        self.nm = root / "nm"
        self.nm.write_text(textwrap.dedent(f"""\
            #!{sys.executable}
            import sys
            name = "demangled.txt" if "-C" in sys.argv else "raw.txt"
            sys.stdout.write(open({str(root)!r} + "/" + name).read())
            """))
        self.nm.chmod(self.nm.stat().st_mode | stat.S_IEXEC)
        self.allow = root / "allow.txt"
        self.allow.write_text(allow)
        self.obj = root / "object.o"
        self.obj.write_bytes(b"")

    def run(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = check_shared_symbols.main(["check", str(self.nm), str(self.obj), str(self.allow)])
        return code, out.getvalue()

    def close(self):
        self.dir.cleanup()


class CheckSharedSymbols(unittest.TestCase):
    def check(self, demangled, allow, raw=FUNCTIONS):
        fixture = Fixture(demangled, allow, raw)
        try:
            return fixture.run()
        finally:
            fixture.close()

    def test_all_allowed_passes(self):
        code, out = self.check("0000 u Table::kRows\n0000 V std::thing\n",
                               "Table::kRows  # constant table\nstd::.*  # the library's own\n")
        self.assertEqual(code, 0, out)
        self.assertIn("2 allowed, 0 not", out)

    def test_unlisted_symbol_fails_and_is_named(self):
        code, out = self.check("0000 u Table::kRows\n0000 u counter()::value\n", "Table::kRows  # constant table\n")
        self.assertEqual(code, 1)
        self.assertIn("one copy per module on Windows: counter()::value", out)

    def test_every_shared_type_counts(self):
        for kind in ("u", "V", "v", "B", "D", "G", "S"):
            with self.subTest(kind):
                code, out = self.check(f"0000 {kind} g_state\n", "other  # unused\n")
                self.assertEqual(code, 1, out)
                self.assertIn("g_state", out)

    def test_functions_and_local_symbols_are_not_data(self):
        code, out = self.check("0000 W inline_function()\n0000 T f()\n0000 r local_table\n0000 b local_state\n",
                               "unused  # keeps the list non-empty\n")
        self.assertIn("0 shared data symbols", out)

    def test_stale_entry_fails(self):
        code, out = self.check("0000 u Table::kRows\n", "Table::kRows  # constant table\nGone::x  # removed long ago\n")
        self.assertEqual(code, 1)
        self.assertIn("matches nothing any more: Gone::x", out)

    def test_pattern_matches_the_whole_name(self):
        # "Table::k" must not allow "Table::kRows": a prefix is not a match.
        code, out = self.check("0000 u Table::kRows\n", "Table::k  # too short\n")
        self.assertEqual(code, 1)
        self.assertIn("Table::kRows", out)

    def test_object_without_inline_functions_fails(self):
        code, out = self.check("0000 u Table::kRows\n", "Table::kRows  # constant table\n", raw="0000 T main\n")
        self.assertEqual(code, 1)
        self.assertIn("-fkeep-inline-functions", out)

    def test_entry_without_reason_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.check("0000 u Table::kRows\n", "Table::kRows\n")
        self.assertIn("needs '  # <why", str(caught.exception))

    def test_repo_allow_list_is_well_formed(self):
        entries = check_shared_symbols.read_allow_list(
            Path(__file__).resolve().parent / "shared_state_allowlist.txt")
        self.assertGreater(len(entries), 5)


if __name__ == "__main__":
    unittest.main()
