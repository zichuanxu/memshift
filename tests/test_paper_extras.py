"""The overview table, worked example, model table and setup facts (`analysis.paper_extras`):
the overview prints the values the paper's macros give for the same quantities, the example's
generator side matches the committed table and its run side reads the raw outputs' keys, and the
facts follow from the committed run metadata. Synthetic inputs check the GPU-hour and spend
rules."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import paper_extras as pe
from memshift.analysis.paper_data import MISSING
from memshift.analysis.paper_tables import _num

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper" / "generated"
SOURCES = json.loads((ROOT / "results" / "sources.json").read_text())
BRANCH_TAGS = (("src", "Src"), ("def", "Def"), ("orphan", "Orphan"))


def _macros() -> dict[str, str]:
    text = (GENERATED / "numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", text))


def _signed(s: str) -> list[str]:
    """The numbers of a printed estimate or interval, with their typeset signs."""
    return [sign + n for sign, n in re.findall(r"(\$[+-]\$)?(\d+\.\d+)", s)]


@pytest.fixture(scope="module")
def overview() -> dict[tuple[str, str, str, str], str]:
    return pe.overview_values(pe.load(ROOT, SOURCES))


@pytest.mark.paper
def test_overview_levels_equal_the_macros(overview: dict[tuple[str, str, str, str], str]) -> None:
    macros = _macros()
    main = {
        "zero": "Zero",
        "rag": "Rag",
        "oracle": "Oracle",
        "holdmem_D0B0": "HoldZeroZero",
        "holdmem_D1B0": "HoldOneZero",
        "holdmem_D0B1_source": "HoldZeroOne",
        "holdmem_D1B1_source": "HoldOneOne",
    }
    for b, bt in BRANCH_TAGS:
        for m, mt in (("Update", "Update"), ("Stability", "Stab")):
            for system, tag in main.items():
                macro = macros[f"ExZeroMain{bt}All{mt}{tag}Est"]
                assert overview["main", system, b, m] + r"\%" == macro
            # the native systems' block: Mem0 and the white-box cells on the three-month worlds
            assert overview["native", "mem0", b, m] + r"\%" == macros[f"EFiveMem{m}{bt}Est"]
            for system, tag in (
                ("holdmem_D0B0", "DzeroBzero"),
                ("holdmem_D1B1_source", "DoneBone"),
            ):
                assert overview["native", system, b, m] + r"\%" == macros[f"EFiveEx{tag}{m}{bt}"]


@pytest.mark.paper
def test_overview_interactions_equal_the_macros(
    overview: dict[tuple[str, str, str, str], str],
) -> None:
    macros = _macros()
    for b, bt in BRANCH_TAGS:
        assert _signed(overview["families", "main", b, "BxD"]) == _signed(macros[f"EOneBxD{bt}"])
    for key, tag in (("llama", "Llama"), ("gemma", "Gemma"), ("gptoss", "GptOss")):
        for b, bt in BRANCH_TAGS[:2]:
            got = _signed(overview["families", key, b, "BxD"])
            assert got == _signed(macros[f"MTwo{tag}BxD{bt}"])


@pytest.mark.paper
def test_overview_native_levels_equal_the_native_analysis(
    overview: dict[tuple[str, str, str, str], str],
) -> None:
    """Every native value of the overview is the native estimate of the native-system analysis,
    against which the native-system tables count and pair (they print it only here)."""
    systems = pe.load(ROOT, SOURCES)["n"]["explore_native_null"]["systems"]
    for key, _ in pe.NATIVE_SYSTEMS:
        for b in ("src", "def", "orphan"):
            for m in ("Update", "Stability"):
                est = systems[key]["comparisons"][f"{b}:{m}"]["native"]["est"]
                assert _num(est) == overview["native", key, b, m], (key, b, m)
    text = (GENERATED / "tab_native_null.tex").read_text()
    assert r"\label{tab:native-diff}" in text and "Native $-$ cell" in text


@pytest.mark.paper
def test_the_results_text_states_the_native_block_item_counts() -> None:
    """The Results' note on the overview's native block (moved there from the caption): the rows
    are scored on the half of the items not used to infer a label, they share the worlds, the native
    systems have fewer item-times than the cells after a deletion only, and the baselines on
    Stability only."""
    n = pe.load(ROOT, SOURCES)["n"]
    for key, _ in pe.NATIVE_SYSTEMS:
        for comp, c in n["explore_native_null"]["systems"][key]["comparisons"].items():
            refs = c["references"]
            cells = refs["holdmem_D0B0"]["n_items"]
            assert (c["native"]["n_items"] < cells) == comp.startswith("orphan:")
            for base in ("zero", "rag", "oracle"):
                assert (refs[base]["n_items"] < cells) == comp.endswith(":Stability")
    text = " ".join((ROOT / "paper" / "sections" / "results.tex").read_text().split())
    assert "scored on the half of the items not used to infer the systems' labels" in text
    assert (
        "share the worlds but not all scored items: the native systems have fewer after a" in text
    )
    assert "baselines fewer on Stability" in text
    caption = pe.tab_overview(pe.load(ROOT, SOURCES)).split(r"\label{tab:overview}")[0]
    assert "share the worlds" not in caption


def test_overview_caption_names_its_rows_in_plain_words() -> None:
    text = pe.tab_overview(pe.load(ROOT, SOURCES))
    caption = text.split(r"\label{tab:overview}")[0]
    assert caption.startswith(
        r"\begin{table*}[t]" + "\n" + r"\caption{Update and Stability (\%) after"
    )
    notes = text.split(r"\end{mttabular}")[1]  # the definitions, under the table
    assert r"$D_{\mathrm{m}}$: the writer's programs." in notes
    assert r"Brackets: 95\% interval." in notes
    assert r"\mtgroup{7}{$B{\times}D$ on Update (pp), each model as writer and reader}" in text
    for gone in ("every evaluated memory", "model-written", "measurement half", "Native systems:"):
        assert gone not in caption + notes, gone


def test_overview_families_block_has_its_own_banner_and_centred_intervals() -> None:
    """The bottom block prints B x D, not a level: its banner states the quantity and unit, and
    each estimate with its interval is one cell centred under its shift's two columns."""
    lines = pe.tab_overview(pe.load(ROOT, SOURCES)).splitlines()
    head = lines.index(r"\mtgroup{7}{$B{\times}D$ on Update (pp), each model as writer and reader}")
    assert lines[head - 1] == r"\midrule"
    rows = lines[head + 1 : head + 1 + 1 + len(pe.FAMILIES)]
    labels = [r.split(r" \mtmuted")[0].removeprefix(r"\mtin ") for r in rows]
    assert labels == ["Qwen3.5-9B", *(f.label for f in pe.FAMILIES)]
    pair = re.compile(
        r"\\multicolumn\{2\}\{c(@\{\})?\}\{\\mtpair\{(\$[+-]\$\d+\.\d|0\.0)\}\{\[.+\]\}\}"
    )
    for row in rows:
        cells = row.removesuffix(r" \\").split(" & ")
        assert len(cells) == 4, row
        assert re.search(r"\\mtmuted\{\(\d+ worlds\)\}$", cells[0]), cells[0]
        for cell in cells[1:]:
            assert pair.fullmatch(cell), cell
        assert cells[3].startswith(r"\multicolumn{2}{c@{}}")


def test_overview_without_stages_prints_the_marker() -> None:
    text = pe.tab_overview({"e1": None, "c": None, "n": None, "m2": {}})
    assert MISSING in text
    assert r"\label{tab:overview}" in text and text.count(r"\\") > 20


@pytest.mark.paper
def test_committed_overview_and_model_table_are_current() -> None:
    assert (GENERATED / "tab_overview.tex").read_text() == pe.HEAD + pe.tab_overview(
        pe.load(ROOT, SOURCES)
    )
    runs = pe.load_runs(ROOT / "runs")
    text = pe.tab_models(runs)
    assert (GENERATED / "tab_models.tex").read_text() == pe.HEAD + text
    qwen = next(ln for ln in text.splitlines() if ln.startswith(r"\mbox{Qwen/Qwen3.5-9B}"))
    roles = "writer, reader, program writer, record rewriter, field extractor, native systems' LLM"
    assert qwen.endswith(f"& {roles} \\\\")
    assert r"\mbox{Claude Opus 5.5}" in text and "ClaudeCalls" not in text
    # one list of models, the Claude models among them under their names
    for gone in ("open weights", "API", "claude-opus-5-5", "claude-sonnet-5-5"):
        assert gone not in text, gone
    # one identifier per model and no revision, serving version or API header date
    shown = text.replace(r"\label{tab:model-revisions}", "")
    for gone in ("c202236235762e1c871ad0ccb60c8ee5ba337b9a", "0.30.0", "2023-06-01", "revision"):
        assert gone not in shown, gone
    for label in ("tab:models", "tab:family-models", "tab:model-revisions"):
        assert rf"\label{{{label}}}" in text


def _model_run(models: dict[str, str], served: dict[str, int], cells: list[str]) -> pe.Run:
    servers = {role: {"model": models[role], "max_model_len": n} for role, n in served.items()}
    entries = {role: {"model": m} for role, m in models.items()}
    return pe.Run("r", "E1", _t(0), _t(1), entries, servers, {"cells": cells})


def test_model_table_lists_size_context_reasoning_and_roles() -> None:
    qwen, emb, oss = "Qwen/Qwen3.5-9B", "Qwen/Qwen3-Embedding-0.6B", "openai/gpt-oss-20b"
    runs = [
        _model_run(
            {"writer": qwen, "embedding": emb},
            {"writer": 32768, "embedding": 8192},
            ["holdmem_D1sB1_source"],
        ),
        _model_run({"writer": qwen, "reader": qwen}, {"writer": 65536, "reader": 65536}, []),
        _model_run({"writer": oss}, {"writer": 16384}, []),
        _model_run({"defwriter": "claude-opus-5-5", "program_writer": "claude-opus-5-5"}, {}, []),
    ]
    text = pe.tab_models(runs)
    rows = [ln for ln in text.splitlines() if ln.startswith(r"\mbox{")]
    assert [r.split("}")[0].removeprefix(r"\mbox{") for r in rows] == [
        qwen,
        "Claude Opus 5.5",
        oss,
        emb,
    ]  # generative before embedders, each by its number of runs, then by name
    assert r"& 9B & 32,768--65,536 &  & writer, reader, program writer \\" in rows[0]
    assert r"& 20B & 16,384 & low & writer \\" in rows[2]
    assert r"& 0.6B & 8,192 &  & embedder \\" in rows[3]
    opus = "high for definition items, low for programs"
    assert rows[1].endswith(
        rf"& \mtna & 1,000,000 & {opus} & definition-item writer, program writer \\"
    )  # the context of Anthropic's models overview; no size stated (a centred dash)
    assert r"\citep{anthropic_models_overview}" in rows[1]
    assert r"\citep{qwen35_2026,hf_qwen3_5_9b}" in rows[0]
    notes = text.split(r"\end{mttabular}")[1]  # the serving setup, under the table
    assert "gpt-oss-20b in MXFP4" in notes  # weights that are not bfloat16
    assert r"\citep{kwon2023vllm}" in notes and "think adaptively" in notes


def test_model_size_reads_the_repository_id() -> None:
    sizes = {
        "Qwen/Qwen3.5-9B": "9B",
        "Qwen/Qwen3-Embedding-0.6B": "0.6B",
        "meta-llama/Llama-3.1-8B-Instruct": "8B",
        "mistralai/Ministral-3-14B-Instruct-2512-BF16": "14B",
        "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B": "9B",
        "openai/gpt-oss-20b": "20B",
        "BAAI/bge-m3": pe.SIZES["BAAI/bge-m3"],
        "claude-sonnet-5-5": "---",
    }
    assert {m: pe.model_size(m) for m in sizes} == sizes


def test_run_roles_add_the_program_writer_and_the_extraction_models() -> None:
    def run(models: dict[str, str], params: dict[str, Any]) -> pe.Run:
        entries = {role: {"model": m} for role, m in models.items()}
        return pe.Run("r", "E1", _t(0), _t(1), entries, {}, params)

    plain = run({"writer": "w", "reader": "w"}, {"cells": ["holdmem_D1B1_source"]})
    assert pe.run_roles(plain) == {"writer": {"writer"}, "reader": {"reader"}}
    dm = run({"writer": "w"}, {"cells": ["holdmem_D0B0", "holdmem_D1sB1_source"]})
    assert pe.run_roles(dm) == {"writer": {"writer", "program_writer"}}
    named = run({"writer": "w", "program_writer": "p"}, {"cells": ["holdmem_D1sB1_source"]})
    assert pe.run_roles(named)["writer"] == {"writer"}  # the programs have their own writer
    extract = {"rewriter": "w", "extractor": "w"}
    both = run({"writer": "w", "reader": "w"}, {"cells": ["holdmem_D1sxB0"], "extraction": extract})
    assert pe.run_roles(both)["writer"] == {"writer", "program_writer", "rewriter", "extractor"}
    assert pe.run_roles(both)["reader"] == {"reader"}  # one role of the model plays them
    other = run({"reader": "x", "embedding": "e"}, {"extraction": {"rewriter": "x"}})
    assert pe.run_roles(other)["reader"] == {"reader", "rewriter"}
    assert pe.run_roles(run({"writer_0": "w"}, {})) == {"writer_0": {"writer"}}


# ---------------------------------------------------------------------------- worked example
@pytest.fixture(scope="module")
def gen() -> dict[str, Any]:
    return pe.example_inputs()


def test_example_item_changes_with_the_definition(gen: dict[str, Any]) -> None:
    pre, post = gen["probes"]
    assert pre["kappa_version"] != post["kappa_version"]
    assert pre["gold_value"] != post["gold_value"]
    assert {oid for oid, _, _ in gen["records"]} <= {o for p in gen["probes"] for o, _ in p["deps"]}


@pytest.mark.paper
def test_committed_example_matches_the_generator(gen: dict[str, Any]) -> None:
    text = (GENERATED / "example_world.tex").read_text()
    for p in gen["probes"]:
        assert pe._gold(p) in text
        v = p["kappa_version"]
        assert f"{v}: {pe._tex(gen['definitions'][v])}" in text
        assert f"probe on {pe._day(p['as_of'], year=True)}" in text
    for oid, _, body in gen["records"]:
        f = pe._fields(body)
        cells = (f["Project"], f["Task"], f["Status"], pe._day(f["Last updated"]))
        assert " & ".join(pe._tex(c) for c in cells) + " & " in text
        assert pe._tex(oid.split(":")[-1]) not in text  # records by their fields, not their ids
    # rendered for the reader: no record ids with revisions, clock times or serialised records
    for gone in ("tasks:", r"$\cdot$", "Priority", "test world 1000", "dependency set &"):
        assert gone not in text, gone
    assert not re.search(r"\d{1,2}:\d{2}|\br\d\b", text)


def _raw_run(tmp: Path, gen: dict[str, Any], gold: Any = None) -> Path:
    ex = pe.EXAMPLE
    d = tmp / ex.run
    d.mkdir(parents=True)
    t0, t1 = gen["times"]
    probes = [dict(p) for p in gen["probes"]]
    if gold is not None:
        probes[1]["gold_value"] = gold
    base = {"world_id": gen["world_id"], "branch": ex.branch}
    rows = [{**p, "tags": {}} for p in probes]
    (d / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    answers = []
    for cell, _ in ex.cells:
        for t, (status, value, ok) in zip(
            (t0, t1),
            ((("answer", ["A & B"], True), ("abstain", None, False))),
            strict=True,
        ):
            answers.append(
                {
                    **base,
                    "system": cell,
                    "fact_id": ex.slot,
                    "as_of": t,
                    "status": status,
                    "value": value,
                    "contract_valid": ok,
                }
            )
    (d / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in answers))
    s_k = '[header] Statement. Evidence: a r1: "x_1"; b r2: "y"; c r1: "z"'
    items = [
        {
            **base,
            "system": cell,
            "slot_id": ex.slot,
            "s_k": s_k,
            "history": [{"ts": t0, "event": "derive"}]
            + (
                [{"ts": t1, "event": "invalidate"}, {"ts": t1, "event": "repair"}]
                if "B1" in cell
                else []
            ),
        }
        for cell, _ in ex.cells
    ]
    (d / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    return tmp


def test_example_reads_the_raw_outputs(tmp_path: Path, gen: dict[str, Any]) -> None:
    raw = pe.example_raw(_raw_run(tmp_path, gen), gen)
    assert raw is not None
    text = pe.example_world(gen, raw)
    before, after = r"\multicolumn{2}{P{5.9cm}}", r"\multicolumn{2}{P{5.0cm}}"
    did = r"\multicolumn{2}{P{\dimexpr2\exG+2\tabcolsep\relax}@{}}"
    assert rf"{before}{{A \& B \pass}} & {after}{{abstains \fail}} & {did}{{kept}} \\" in text
    # the two B = 1 cells
    assert text.count(rf"{after}{{abstains \fail}} & {did}{{invalidated, re-derived}} \\") == 2
    assert "``Statement.''" in text and "Evidence" not in text  # the stored item's first sentence
    # in the column before the fork
    assert rf"stored item & {before}{{``Statement.''}} & {after}{{}} & {did}{{}} \\" in text
    assert MISSING not in text


def test_example_helpers_render_dates_fields_and_the_statement() -> None:
    assert pe._day("2026-06-28 11:30") == "Jun 28"
    assert pe._day("2026-10-08T00:00", year=True) == "Oct 8, 2026"
    assert pe._fields("Task: A: b\nProject: C\nno field") == {"Task": "A: b", "Project": "C"}
    s_k = "[the set | written 2026-09-30] As of 2026-09-30, it is A, B. Then C. Evidence: x"
    assert pe._statement(s_k) == "As of 2026-09-30, it is A, B."
    assert pe._did([]) == "kept" and pe._did(["invalidate"]) == "invalidated"
    assert pe._answer({"status": "answer", "value": ["X"], "contract_valid": True}, bold=True) == (
        r"\textbf{X} \pass"
    )


def test_example_refuses_a_run_whose_gold_differs(tmp_path: Path, gen: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="not the generator's"):
        pe.example_raw(_raw_run(tmp_path, gen, gold=["Other"]), gen)


def test_example_without_raw_outputs_prints_the_marker(tmp_path: Path, gen: dict[str, Any]) -> None:
    assert pe.example_raw(tmp_path, gen) is None
    text = pe.example_world(gen, None)
    assert text.count(MISSING) == 1 + 3 * len(pe.EXAMPLE.cells)  # stored item; answers, action
    assert pe._value(gen["probes"][0]["gold_value"]) in text


# ------------------------------------------------------------------------------ setup facts
def _t(h: float) -> datetime:
    return datetime(2026, 10, 1) + (datetime(2026, 10, 1, 1) - datetime(2026, 10, 1)) * h


def _run(name: str, experiment: str, start: float, end: float, servers: dict[str, Any]) -> pe.Run:
    return pe.Run(name, experiment, _t(start), _t(end), {}, servers)


def test_gpu_hours_hold_union_and_shared_activity() -> None:
    small = {
        "writer": {"model": "m", "gpus": "0", "port": 1, "launched_utc": "2026-10-01T00:00:00Z"}
    }
    big = {
        "writer": {"model": "b", "gpus": "0,1", "port": 2, "launched_utc": "2026-10-01T05:00:00Z"}
    }
    runs = [
        _run("a", "E1", 0.5, 2, small),
        _run("b", "E5", 1, 3, small),  # shares GPU 0 with run a from 1 to 2
        _run("c", "E2", 5, 6, big),  # two GPUs for one hour
    ]
    h = pe.gpu_hours(runs)
    assert h["total"] == pytest.approx(3 + 2)  # server held 0-3 on GPU 0, 5-6 on GPUs 0 and 1
    assert h["other"] == pytest.approx(0.5 + 0.5 + 2)
    assert h["native"] == pytest.approx(0.5 + 1)


def test_claude_usage_prices_online_calls_and_batch_requests() -> None:
    calls = [
        {
            "key": "a",
            "model": "claude-sonnet-5-5",
            "cached": False,
            "prompt_tokens": 10**6,
            "completion_tokens": 0,
        },
        {
            "key": "a",
            "model": "claude-sonnet-5-5",
            "cached": False,
            "prompt_tokens": 10**6,
            "completion_tokens": 0,
        },  # sent online twice: paid twice
        {
            "key": "b",
            "model": "claude-opus-5-5",
            "cached": True,
            "prompt_tokens": 0,
            "completion_tokens": 10**6,
        },  # never online: one batch request
        {
            "key": "b",
            "model": "claude-opus-5-5",
            "cached": True,
            "prompt_tokens": 0,
            "completion_tokens": 10**6,
        },
        {
            "key": "a",
            "model": "claude-sonnet-5-5",
            "cached": True,
            "prompt_tokens": 10**6,
            "completion_tokens": 0,
        },  # a cache hit of an online call
        {
            "key": "q",
            "model": "Qwen/Qwen3.5-9B",
            "cached": False,
            "prompt_tokens": 10**6,
            "completion_tokens": 10**6,
        },
    ]
    u = pe.claude_usage(calls)
    sonnet_in, _ = pe.BATCH_PRICE["claude-sonnet-5-5"]
    _, opus_out = pe.BATCH_PRICE["claude-opus-5-5"]
    assert u["tokens"] == 3 * 10**6
    assert u["usd"] == pytest.approx(2 * 2 * sonnet_in + opus_out)


@pytest.mark.paper
def test_setup_facts_from_the_committed_runs() -> None:
    runs = pe.load_runs(ROOT / "runs")
    text = pe.setup_facts(runs, None)
    facts = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", text))
    assert facts["SonnetModelId"] == r"\texttt{claude-sonnet-5-5}"
    assert facts["OpusModelId"] == r"\texttt{claude-opus-5-5}"
    assert (facts["ClaudeCallsFrom"], facts["ClaudeCallsTo"]) == ("2026-10-16", "2026-10-19")
    assert facts["ClaudeTokensTotal"] == facts["ClaudeSpendUSD"] == MISSING  # no raw calls here
    committed = dict(
        re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (GENERATED / "setup_facts.tex").read_text()
        )
    )
    for name, value in facts.items():
        if value != MISSING:
            assert committed[name] == value
    hours = [int(facts[k]) for k in ("GpuHoursTotal", "GpuHoursOtherRuns", "GpuHoursNativeSystems")]
    assert hours[0] >= hours[1] + hours[2] - 1  # the families share the active time
