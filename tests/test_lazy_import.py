"""Lazy-import behavior: matplotlib as the optional ``[plot]`` extra.

The in-process blocker tests (not subprocesses) are what keep the error
branch of ``pybenford.__getattr__`` inside coverage measurement.
"""

from __future__ import annotations

import importlib.abc
import importlib.metadata
import re
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

import pybenford

PLOT_NAMES = (
    "plot_digit_test",
    "plot_distortion_factor",
    "plot_mantissa_arc",
    "plot_ordered_mantissas",
    "plot_summation",
    "plot_z_scores",
)

_SENTINEL = object()


class _BlockingFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that raises ModuleNotFoundError for blocked modules."""

    def __init__(self, blocked_prefixes: Sequence[str], error_name: str) -> None:
        self.blocked_prefixes = tuple(blocked_prefixes)
        self.error_name = error_name

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in self.blocked_prefixes
        ):
            raise ModuleNotFoundError(f"No module named {self.error_name!r}", name=self.error_name)
        return None


@pytest.fixture
def block_import() -> Iterator[Callable[[Sequence[str], str], None]]:
    """Prepare a cold-ish import state and let the test install a blocker.

    Setup (i) removes ``matplotlib*`` and ``pybenford.visualization`` from
    ``sys.modules``, (ii) also saves and pops
    ``pybenford.__dict__["visualization"]`` — pytest collection has already
    bound the submodule as a package attribute, and
    ``from pybenford import visualization`` would return that stale attribute
    without ever consulting the finder — and (iv) clears the lazy cache.
    The yielded callable performs (iii): install the meta-path blocker.
    Teardown restores sys.modules, the package attribute, and the cache.
    """

    def _is_blocked_module(mod_name: str) -> bool:
        return (
            mod_name == "matplotlib"
            or mod_name.startswith("matplotlib.")
            or mod_name == "pybenford.visualization"
        )

    # (i) remove matplotlib* and pybenford.visualization from sys.modules
    saved_modules = {
        mod_name: sys.modules.pop(mod_name)
        for mod_name in list(sys.modules)
        if _is_blocked_module(mod_name)
    }
    # (ii) pop the submodule bound as a package attribute at collection time
    saved_pkg_attr = pybenford.__dict__.pop("visualization", _SENTINEL)
    # (iv) clear the lazy cache so __getattr__ runs
    saved_cache = {
        name: pybenford.__dict__.pop(name) for name in PLOT_NAMES if name in pybenford.__dict__
    }

    finders: list[_BlockingFinder] = []

    def _install(blocked_prefixes: Sequence[str], error_name: str) -> None:
        # (iii) install the meta-path blocker
        finder = _BlockingFinder(blocked_prefixes, error_name)
        finders.append(finder)
        sys.meta_path.insert(0, finder)

    yield _install

    for finder in finders:
        sys.meta_path.remove(finder)
    for mod_name in list(sys.modules):
        if _is_blocked_module(mod_name):
            del sys.modules[mod_name]
    sys.modules.update(saved_modules)
    if saved_pkg_attr is _SENTINEL:
        pybenford.__dict__.pop("visualization", None)
    else:
        pybenford.__dict__["visualization"] = saved_pkg_attr
    for name in PLOT_NAMES:
        pybenford.__dict__.pop(name, None)
    pybenford.__dict__.update(saved_cache)


def test_missing_matplotlib_raises_helpful_import_error(
    block_import: Callable[[Sequence[str], str], None],
) -> None:
    """(a) Missing matplotlib is relabeled with the pybenford[plot] hint."""
    block_import(("matplotlib",), "matplotlib")
    with pytest.raises(ImportError, match=r"pybenford\[plot\]") as excinfo:
        pybenford.__getattr__("plot_digit_test")
    assert "pybenford[plot]" in str(excinfo.value)
    assert "plot_digit_test" in str(excinfo.value)


def test_unrelated_import_failure_propagates_unchanged(
    block_import: Callable[[Sequence[str], str], None],
) -> None:
    """(b) A non-matplotlib ModuleNotFoundError is never relabeled."""
    block_import(("pybenford.visualization",), "some_other_module")
    with pytest.raises(ModuleNotFoundError) as excinfo:
        pybenford.__getattr__("plot_digit_test")
    assert excinfo.value.name == "some_other_module"
    assert "pybenford[plot]" not in str(excinfo.value)


def test_cold_import_does_not_load_matplotlib() -> None:
    """(c) Importing pybenford alone must not import matplotlib."""
    result = subprocess.run(
        [sys.executable, "-c", "import sys, pybenford; assert 'matplotlib' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cold_lazy_load_success() -> None:
    """(d) Plot access after a cold import loads matplotlib lazily."""
    code = (
        "import sys\n"
        "import pybenford\n"
        "assert 'matplotlib' not in sys.modules\n"
        "assert callable(pybenford.plot_digit_test)\n"
        "assert 'matplotlib' in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_attribute_identity_and_cache() -> None:
    """(e) Lazy attribute is the visualization function, then cached."""
    pybenford.__dict__.pop("plot_digit_test", None)
    assert pybenford.plot_digit_test is pybenford.visualization.plot_digit_test
    # cached: the second access finds it in pybenford.__dict__
    assert "plot_digit_test" in pybenford.__dict__
    assert pybenford.__dict__["plot_digit_test"] is pybenford.plot_digit_test


def test_unknown_attribute_raises_attribute_error() -> None:
    """(f) Unknown attributes raise AttributeError naming the attribute."""
    with pytest.raises(AttributeError, match="nonexistent_name"):
        _ = pybenford.nonexistent_name


def test_dir_lists_plot_names_before_any_plot_access() -> None:
    """(g) dir() advertises all six plot names even with an empty cache."""
    for name in PLOT_NAMES:
        pybenford.__dict__.pop(name, None)
    listed = dir(pybenford)
    for name in PLOT_NAMES:
        assert name in listed


def test_distribution_metadata_matches_declared_extras() -> None:
    """(h) Installed metadata: matplotlib only behind extras, numpy/scipy core."""
    requires = importlib.metadata.requires("pybenford")
    assert requires is not None

    def req_name(req: str) -> str:
        match = re.match(r"[A-Za-z0-9._-]+", req)
        assert match is not None, req
        return match.group(0)

    normalized = [req.replace("'", '"') for req in requires]

    # (i) no unconditional matplotlib; both plot and dev extra entries exist
    matplotlib_entries = [r for r in normalized if req_name(r) == "matplotlib"]
    assert matplotlib_entries, normalized
    assert all(";" in r and "extra ==" in r for r in matplotlib_entries)
    assert any('extra == "plot"' in r for r in matplotlib_entries)
    assert any('extra == "dev"' in r for r in matplotlib_entries)

    # (ii) exactly one unconditional numpy (>=1.22.0) plus the bounded dev pin
    numpy_entries = [r for r in normalized if req_name(r) == "numpy"]
    unconditional_numpy = [r for r in numpy_entries if ";" not in r]
    assert len(unconditional_numpy) == 1, numpy_entries
    assert ">=1.22.0" in unconditional_numpy[0]
    dev_numpy = [r for r in numpy_entries if 'extra == "dev"' in r]
    assert len(dev_numpy) == 1, numpy_entries
    assert ">=2.2" in dev_numpy[0]
    assert "<2.5" in dev_numpy[0]

    # (iii) an unconditional scipy entry
    scipy_entries = [r for r in normalized if req_name(r) == "scipy"]
    assert any(";" not in r for r in scipy_entries), normalized
