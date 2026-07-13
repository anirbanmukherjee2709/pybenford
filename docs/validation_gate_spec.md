# Spec: Stand Up the Validation Gate (Pass 1 of 3)

**Status:** Rev 4 — third Codex review folded in (machine-enforced version
assertions, min-deps uses frozen pytest pins)
**Author:** Claude.ai planning session, 2026-07-13
**Implementer:** Claude Code (sole author — Codex is suggest-only, never writes)
**Scope:** Tooling, configuration, CI, and two README factual corrections. One
public packaging change: the supported-Python floor moves from 3.9 to 3.10.
ZERO changes to statistical behavior.

## Read-first instructions (mandatory, in order)

1. Read this spec completely before writing anything.
2. Read `pyproject.toml` and `README.md`.
3. Read `src/pybenford/__init__.py`, `src/pybenford/distributions.py` (lines
   140–200), `src/pybenford/utils.py` (lines 36–70), `tests/test_display.py`,
   `tests/test_statistics.py` — these contain every semantic edit this spec
   permits.
4. Establish a trustworthy baseline (Codex finding 1 — the gate must exercise
   THIS checkout, not a stale site-packages copy):

   ```
   pyenv activate pybenford-dev
   pip install -e ".[dev]"
   python -c "import pybenford; p = pybenford.__file__; assert '/projects/pybenford/src/' in p, p; print(p)"
   pytest -q
   ```

   The assert must pass and all 212 tests must be green. If either fails,
   STOP and report — do not proceed.

## Anti-hallucination guardrails

- Do NOT modify any statistical logic, function signature, docstring content,
  public API name, or test assertion semantics.
- Do NOT bump the version or publish to PyPI. Commits only.
- Do NOT fix, refactor, or "improve" anything not enumerated below. Real bugs
  you notice get reported in your summary and route to Pass 2.
- Any dependency-resolution failure, unexpected mypy error, or unexpected ruff
  finding is a STOP-AND-REVIEW condition. Never improvise a fix for a problem
  this spec does not list, and never raise a dependency floor on your own
  (Codex finding 4).
- Every claimed result must come from actually running the command; paste real
  output in the summary.

## Task 1 — Python floor: drop 3.9 (public change)

Python 3.9 is EOL (Oct 2025) and current mypy cannot target it. In
`pyproject.toml`:

- `requires-python = ">=3.10"`
- Remove the `Programming Language :: Python :: 3.9` classifier.
- `[tool.ruff] target-version = "py310"`

This lands in the 0.2.0 CHANGELOG entry in Pass 2; no CHANGELOG edit now.

## Task 2 — mypy strict with a pinned, verified toolchain

Replace the three individual mypy flags:

```toml
[tool.mypy]
python_version = "3.10"
strict = true
```

Pin the type/lint toolchain exactly (Codex findings 2 and 6, plus Rev 2
blockers 1 and 3). Verified combination — with these pins a Python 3.10
interpreter yields exactly the 4 errors listed below:

```toml
dev = [
    "pytest==9.1.1",
    "pytest-cov==7.1.0",
    "ruff==0.15.21",
    "mypy==2.2.0",
    "scipy-stubs==1.15.3.0",
    "optype==0.9.3",
    "pre-commit>=3.0",
    "build>=1.0",
    "twine>=5.0",
]
```

`optype` is a gate-affecting TRANSITIVE dependency of scipy-stubs and must be
pinned explicitly: unpinned, Python 3.13 (the local `pybenford-dev` env)
resolves optype 0.18.0, whose `type`-statement syntax breaks mypy when
targeting `python_version = "3.10"`. Python 3.10 naturally resolves 0.9.3;
the pin makes local 3.13 and CI 3.10 agree. (`pre-commit`, `build`, `twine`
float — their output is not part of the gate.)

Also amend `[tool.pytest.ini_options]` to enforce the coverage floor (Rev 2
blocker 4):

```toml
addopts = "--cov=pybenford --cov-report=term-missing --cov-fail-under=99 --strict-markers"
```

Current measured coverage is ~99.4%, so 99 passes with headroom.

Then fix exactly these 4 strict errors (verified against the pins above).
NOTE: the line numbers below are pre-format; the formatter commit runs first,
so locate each site by content, not line number:

- `src/pybenford/distributions.py:166` and `:196` — `no-any-return`. Annotate
  the intermediate, e.g.
  `result: NDArray[np.float64] = np.log10(...).sum(axis=...)` then
  `return result`. Do not change the math.
- `src/pybenford/utils.py:56` and `:58` — remove the two now-unused
  `# type: ignore[union-attr]` comments.

If `mypy src` reports anything not in this list: STOP and report verbatim.

## Task 3 — ruff: pinned, drift fixed, format adopted

With `ruff==0.15.21` (verified to reproduce all findings below):

- Add `"RUF002", "RUF003"` to `[tool.ruff.lint] ignore` — docstring en/em
  dashes are intentional typography.
- Fix the remaining findings:
  - `I001` import sort: `src/pybenford/__init__.py`, `tests/test_display.py`,
    `tests/test_statistics.py`
  - `RUF022` unsorted `__all__`: `src/pybenford/__init__.py`
  - `F401` unused imports: `tests/test_display.py` (pytest),
    `tests/test_statistics.py` (DistortionResult, MADResult)
  - `E741`/`RUF015`/`F841` in `tests/test_display.py` — rename `l` → `line`,
    use `next(...)`, delete the unused `digit_lines` binding. Assertions stay
    semantically identical.
- `ruff format src tests` runs on the OTHERWISE-UNTOUCHED tree as commit 2,
  BEFORE any Task 1–5 edit (see Commit plan — Rev 2 blocker 5 made this
  explicit). Mechanical, whitespace-only reformatting may touch ANY file under
  `src/` and `tests/` (the explicit formatter carve-out from the semantic
  allowlist; ~9 files expected to change).

## Task 4 — README factual corrections (Codex finding 7)

`README.md` currently claims 210 tests and 100% coverage. Actual, verified:
212 tests, 99% coverage. Correct both numbers. No other README edits.

## Task 5 — CI workflow (prescribed skeleton, Codex findings 2, 4, 5)

Create `.github/workflows/ci.yml` with exactly this structure (adjust only if
actionlint flags an error, and report any adjustment):

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  lint-type:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: python -m pip install --upgrade pip
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: mypy src

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pip install --upgrade pip
      - run: pip install -e ".[dev]"
      - run: pytest -q

  min-deps:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: python -m pip install --upgrade pip
      - run: pip install numpy==1.22.0 scipy==1.8.0 matplotlib==3.5.0
      - run: pip install --no-deps -e .
      - run: pip install pytest==9.1.1 pytest-cov==7.1.0
      - run: pip check
      - run: python -c "import numpy, scipy, matplotlib; assert numpy.__version__ == '1.22.0' and scipy.__version__ == '1.8.0' and matplotlib.__version__ == '3.5.0'"
      - run: pytest -q
```

Design notes (do not deviate without reporting):

- mypy runs ONCE in `lint-type` on 3.10 — the pinned scipy-stubs cannot be
  type-checked consistently across a 3.10–3.13 matrix (Codex finding 2). The
  matrix jobs run tests only.
- `min-deps` installs the EXACT declared floors (`==1.22.0`, not `1.22.*`),
  then the package with `--no-deps`, then test tools, then verifies with
  `pip check` and version asserts (Codex finding 4). If the floors fail to
  resolve or tests fail: STOP and report — that is a review decision about
  the declared floors, not something to patch silently.
- Validate the workflow with actionlint
  (`brew install actionlint && actionlint .github/workflows/ci.yml`). If
  actionlint cannot be installed, say so in the summary; first push is then
  the live validation (Codex finding 5).

## Validation gate (mandatory before the final commit)

The baseline install used the OLD unpinned pyproject.toml. After editing the
pins, reinstall and assert the declared toolchain is actually what runs (Rev 2
blocker 2), then run the gate:

```
pip install -e ".[dev]"
python -c "import pybenford; p = pybenford.__file__; assert '/projects/pybenford/src/' in p, p"
python - <<'EOF'
from importlib.metadata import version
pins = {
    "pytest": "9.1.1",
    "pytest-cov": "7.1.0",
    "ruff": "0.15.21",
    "mypy": "2.2.0",
    "scipy-stubs": "1.15.3.0",
    "optype": "0.9.3",
}
bad = {p: (version(p), want) for p, want in pins.items() if version(p) != want}
assert not bad, f"toolchain drift: {bad}"
print("toolchain pins verified")
EOF
ruff check src tests
ruff format --check src tests
mypy src            # strict via pyproject; expect: no errors, 8 files checked
pytest -q           # expect: 212 passed, coverage gate >= 99 enforced
```

The heredoc is a real assertion — it exits non-zero on any drift across all
six pinned tools. If it fails, STOP; do not run the gate against stale tools.
Paste actual output in the summary. Never commit red.

## Commit plan (Codex findings 3 and 8)

Three commits, in this exact execution order (Rev 2 blocker 5 — format runs
on a tree containing no other changes):

1. `docs: add validation gate spec (pass 1)` — this file, committed FIRST,
   before any other change.
2. `style: apply ruff format` — run `ruff format src tests` on the
   otherwise-untouched tree and commit its output only. No manual edits, no
   Task 1–5 changes mixed in.
3. `chore: stand up strict validation gate (ruff pin, mypy strict, CI, README counts)`
   — ALL Task 1–5 changes, made only after commit 2 exists, validated by the
   full gate above.

Done-check uses `git status --porcelain` (must be empty), not `git diff`
(which ignores untracked files).

## Done criteria

- Baseline assert proves the gate ran against this checkout.
- Gate passes locally with pinned toolchain; pins recorded in summary.
- `.github/workflows/ci.yml` matches the skeleton and passes actionlint (or
  the summary states actionlint was unavailable).
- README says 212 tests, 99% coverage.
- `git status --porcelain` is empty after commit 3; history shows the three
  commits above and nothing else.
- Test count still 212, all passing.
