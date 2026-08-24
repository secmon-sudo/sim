"""Every LLM call must name the stage that spent it, 2026-08-24.

system_telemetry counted LLM calls from the day the pipeline started, but the row
carried no stage label and two thirds of the stages never wrote a row at all: only
pass_c and the storyline adjudicator logged. The table could answer "how many calls"
and never "on what", which made the prose stages — the ones on the expensive quality
router — look free.

These tests guard the two ways that regresses: a stage that calls the model without
logging, and a log call with a purpose outside the closed vocabulary.
"""

import ast
import pathlib

import pytest

from src.core.llm_client import PURPOSES

REPO = pathlib.Path(__file__).resolve().parent.parent

# heartbeat.py mentions call_llm only inside a class docstring showing usage.
_NO_CALL_SITES = {"src/core/heartbeat.py"}


def _modules_calling(name: str) -> set[str]:
    """Modules with a real call to `name` — ast, so docstrings don't count."""
    found = set()
    for path in list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if called == name and not isinstance(node.func, ast.Attribute):
                rel = str(path.relative_to(REPO))
                if rel not in _NO_CALL_SITES:
                    found.add(rel)
    return found


def test_every_llm_calling_module_is_instrumented():
    """A module that spends LLM calls must also account for them.

    The accounting may live in the module itself or in its caller, so this asserts
    the pair exists SOMEWHERE for each spending module — the point is that adding a
    new stage cannot silently escape the spend rollup.
    """
    spenders = _modules_calling("call_llm")
    assert spenders, "no call_llm sites found — the AST walk is broken, not the code"

    loggers = _modules_calling("log_llm_telemetry")
    # Stages whose spend is recorded by their caller rather than in-module.
    logged_by_caller = {
        "src/services/sitrep_generator.py": "src/pipeline/daily_sitrep.py",
        "src/pipeline/pass_c_classify.py": "src/pipeline/pass_c_classify.py",
    }
    for module in sorted(spenders):
        accounted = module in loggers or logged_by_caller.get(module) in loggers
        assert accounted, f"{module} calls the model but nothing logs the spend"


def test_purposes_are_unique_and_named():
    assert len(PURPOSES) == len(set(PURPOSES))
    assert all(p and p.islower() and " " not in p for p in PURPOSES)


@pytest.mark.parametrize("expected", [
    "classify_batch", "classify_single", "storyline_narrative",
    "sitrep_country", "sitrep_digest",
    "forecast_g1_selection", "forecast_g2_country", "forecast_g3_global",
])
def test_known_stages_are_in_the_vocabulary(expected):
    assert expected in PURPOSES


def test_every_logged_purpose_is_in_the_vocabulary():
    """No literal passed as purpose= may sit outside PURPOSES."""
    used = set()
    for path in list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if called not in ("log_llm_telemetry", "_log"):
                continue
            for kw in node.keywords:
                if kw.arg == "purpose" and isinstance(kw.value, ast.Constant):
                    used.add(kw.value.value)
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value in PURPOSES:
                    used.add(arg.value)
    assert used, "no purpose literals found — instrumentation is missing"
    assert used <= set(PURPOSES), f"unknown purposes: {sorted(used - set(PURPOSES))}"
