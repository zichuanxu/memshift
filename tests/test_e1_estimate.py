"""The offline call estimate agrees with what the cells send (replayed with recording writers)."""

import pytest

from memshift.experiments.e1 import parse_cells
from memshift.experiments.e1_estimate import Recorder, estimate, summarise_full_access
from memshift.holdmem.writer import THINK_MAX_TOKENS
from tests.e1_replay import replay

TOKENS = (
    "D0B1_gold,D0B1_scope,D0B0_refresh,D0tB0,D0tB1_source,D0thinkB1_source,D1sB1_source_ra,zero_Dm"
)


@pytest.fixture(scope="module")
def result():
    cells, views = parse_cells(TOKENS)
    return estimate([5], ("def", "orphan"), cells, views, months=3)["per_world"]["seed5"]


def test_cells_that_repeat_existing_requests_need_no_new_call(result):
    for name in ("holdmem_D0B0_refresh", "holdmem_D0tB0", "holdmem_D0tB1_source"):
        assert result[name]["writer_new"] == 0 and result[name]["program_new"] == 0, name
    assert result["holdmem_D0thinkB1_source"]["writer_new"] == 0  # the plain part is cached
    assert result["zero_Dm"]["program_new"] == 0 and result["zero_Dm"]["writer_new"] == 0


def test_keys_send_only_the_requests_no_source_change_explains(result):
    for name in ("holdmem_D0B1_gold", "holdmem_D0B1_scope"):
        assert 0 <= result[name]["writer_new"] <= 10, name  # the clock moved a window
        assert result[name]["writer_distinct"] < result["holdmem_D0B0_refresh"]["writer_distinct"]


def test_new_calls_are_the_thinking_and_the_retraction_aware_program_requests(result):
    think = result["holdmem_D0thinkB1_source"]["think"]
    assert 0 < think <= 6 * 4 * 2  # six definition items, four checkpoints, two branches
    ra = result["holdmem_D1sB1_source_ra"]
    assert ra["program_new"] == ra["program_distinct"] > 0
    assert ra["writer_new"] == 0


def test_the_view_runs_once_per_asked_probe(result):
    assert result["zero_Dm"]["view_runs"] == result["zero_Dm"]["reader_upper"] > 0


def test_the_estimate_matches_the_requests_of_a_replay():
    """The estimate's request counts are those a replay with recording fakes sends (the bodies carry
    the model name, so the keys differ, the counts do not)."""
    cells, _ = parse_cells("D0B0,D0B1_gold")
    est = estimate([5], ("def",), cells, (), months=3, reference=[])["per_world"]["seed5"]
    rep = replay(cells, "def", 5, (1, 2, 4, 8))
    for cfg in cells:
        assert est[cfg.name]["writer_distinct"] == len(rep.keys(cfg.name, "write")), cfg.name


def test_a_think_request_differs_in_body_from_the_plain_one():
    plain, think = Recorder(), Recorder({"think": True})
    messages = [{"role": "user", "content": "x"}]
    a = plain.complete(messages, 10, None, "write")
    b = think.complete(messages, 10, None, "write")
    assert a.key != b.key and THINK_MAX_TOKENS > 700


def test_the_estimate_counts_the_new_requests_of_d0full_and_reports_what_its_writer_gets():
    """D0full sends new requests for the definition items after the shift only; the estimate says
    how many, and how large the requests are and how often the cap cuts (what the cell's own item
    notes record)."""
    cells, _ = parse_cells("D0B1_source,D0fullB1_source")
    src, full = cells
    weeks = (1, 2, 4, 8)
    est = estimate([5], ("def",), [full], (), months=3, reference=[src])["per_world"]["seed5"]
    cell = est[full.name]
    rep = replay(cells, "def", 5, weeks)
    new = rep.keys(full.name, "write") - rep.keys(src.name, "write")
    assert cell["writer_new"] == len(new) > 0 and cell["program_new"] == 0 and cell["think"] == 0
    rows = [h for i in rep.cells[full.name].items.values() for h in i.history if "full" in h]
    access = cell["full_access"]
    assert access["rederivations"] == len(rows) > 0
    assert access["truncated"] == sum(h["full"]["dropped"] > 0 for h in rows) > 0  # the e-mail
    assert access["range_incomplete"] <= access["truncated"]
    assert access["newest_first"] == 0  # the fallback rule is not needed at the default cap
    assert set(access["dropped_by_source"]) == {"email"}  # only the e-mail is cut, never the cards
    assert set(access["by_item"]) == {s.kappa_id for s in rep.world.slots if s.kappa_id}
    assert access["prompt_tokens"]["max"] > access["record_tokens"]["max"]
    assert summarise_full_access([]) == {}
