import json
from dataclasses import replace
from datetime import timedelta

import pytest

from memshift.experiments import e1
from memshift.experiments.common import read_jsonl, world_params
from memshift.gen import llm_render, pools
from memshift.gen.fidelity import check_world, surface_forms
from memshift.gen.gold import gold_at
from memshift.gen.llm_render import (
    RENDER_SOURCES,
    RETRY_MARK,
    Anchor,
    RenderFidelityError,
    check_body,
    clean_body,
    event_anchors,
    is_background,
    render_request,
    render_world_llm,
)
from memshift.gen.render import render_text
from memshift.gen.shifts import BRANCHES, generate_branch
from memshift.gen.state import SourceLog
from memshift.runinfo import verify_manifest, world_record
from tests.fakes import FakeLLM

SMALL = {"target_tokens": 30_000}


@pytest.fixture(scope="module")
def worlds():
    return {b: generate_branch(7, b, SMALL) for b in BRANCHES}


def _events(world, source):
    return [e for e in world.events if e.source == source and e.op != "delete"]


def _is_digest(e):
    return e.source == "email" and e.payload["from_addr"].startswith("digest@")


def _body(new, old):
    """The rewritten body: the text after the template header (header lines never change)."""
    header = render_text(old.source, {**old.payload, "body": ""})
    assert new.payload["text"].startswith(header)
    return new.payload["text"][len(header) :]


def test_anchors_of_known_facts(worlds):
    world = worlds["src"]
    anchors = event_anchors(world)
    by_rev = {(e.object_id, e.revision): e for e in world.events}

    def anchored(oid, rev=1):
        return anchors[by_rev[(oid, rev)].event_id]

    # diet: the chat message and the meal-plan note both carry the value, in different forms
    first = world.timelines["u.diet"][0]
    msg_oid, note_oid = first.deps
    forms = surface_forms(world.slot("u.diet"), first.value)
    (msg_anchor,) = anchored(msg_oid)
    assert msg_anchor.shown == (pools.DIET_PHRASES[first.value],)
    assert msg_anchor.alternatives == tuple(tuple(c) for c in forms)
    assert msg_anchor.satisfied(f"I'm now {pools.DIET_PHRASES[first.value]}.")
    assert msg_anchor.satisfied(f"Current diet: {first.value}")  # any alternative satisfies
    assert not msg_anchor.satisfied("current diet: nothing in particular")
    (note_anchor,) = anchored(note_oid)
    assert note_anchor.shown in [
        (f"Current diet: {first.value}",),
        (pools.DIET_PHRASES[first.value],),
    ]
    # employer and manager: the welcome email carries both
    emp, mgr = world.timelines["u.employer"][0], world.timelines["u.manager"][0]
    assert emp.deps == mgr.deps
    shown = {a.shown for a in anchored(emp.deps[0])}
    assert (emp.value,) in shown and (mgr.value,) in shown
    # working hours: the sentence the evaluator's regex reads
    wh = [e for e in _events(world, "notes") if e.payload["title"] == "Working hours"]
    assert wh
    for e in wh:
        (a,) = anchors[e.event_id]
        assert a.shown[0] in e.payload["body"] and a.shown[0].startswith("My working hours: ")
    # emails about appointments are not gold dependencies: no anchor
    confirmations = [
        e for e in _events(world, "email") if e.payload["subject"] == "Appointment confirmation"
    ]
    assert confirmations and all(e.event_id not in anchors for e in confirmations)


def test_anchored_events_are_dependencies_or_evaluator_phrases(worlds):
    world = worlds["natural"]
    dep_oids = {oid for pts in world.timelines.values() for p in pts for oid in p.deps}
    by_id = {e.event_id: e for e in world.events}
    assert event_anchors(world)
    for eid in event_anchors(world):
        e = by_id[eid]
        assert e.object_id in dep_oids or e.payload.get("title") == "Working hours"


def test_conflict_values_are_anchors(worlds):
    world = worlds["conflict"]
    anchors = event_anchors(world)
    by_oid = {e.object_id: e for e in world.events}
    assert len(world.params["conflicts"]) == 4
    for c in world.params["conflicts"].values():
        msg = by_oid[c["objects"][1]]
        assert msg.source in RENDER_SOURCES
        assert (c["values"][1],) in {a.shown for a in anchors[msg.event_id]}


def test_kappa_member_names_are_anchors_when_the_body_carries_them(worlds):
    world = worlds["natural"]
    by_email = {p.email: p.name for p in world.people.values()}
    chat = _events(world, "messages")[0]
    mail = next(e for e in _events(world, "email") if e.payload["from_addr"] in by_email)
    names = {chat.event_id: chat.payload["conversation_with"]}
    names[mail.event_id] = by_email[mail.payload["from_addr"]]
    edited = [
        replace(e, payload={**e.payload, "body": f"Thanks {names[e.event_id]}, see you soon."})
        if e.event_id in names
        else e
        for e in world.events
    ]
    anchors = event_anchors(replace(world, events=edited))
    for eid, name in names.items():
        assert {a.shown for a in anchors[eid]} == {(name,)}
    assert not set(names) & set(event_anchors(world))  # the template bodies name no one


def test_only_free_text_sources_can_be_rendered(worlds):
    with pytest.raises(ValueError):
        render_world_llm(worlds["natural"], FakeLLM(), sources=("email", "calendar"))


def test_guards_reject_additions_and_missing_anchors():
    tpl = "Hi Avery,\n\nYour visit is on March 03, 2026 at 09:00 in Berlin. Bring a coat.\n\nDana"
    keep = (Anchor((("Berlin",),), ("Berlin",)),)

    def kinds(body):
        return [k for k, _ in check_body(body, tpl, keep)]

    ok = "Hi Avery, see you March 3, 2026 at 9:00 in Berlin - Dana. Young people love it."
    assert kinds(ok) == []  # reformatted date accepted, common-word surname ignored
    assert kinds("Hi Avery, see you in berlin") == ["anchor"]  # case matters
    assert kinds("Hi Avery, see you soon") == ["anchor"]
    assert kinds("Berlin at 15:30") == ["number"]
    assert kinds("Berlin " + "7" * 5000) == ["number"]  # no int() digit cap
    assert kinds("Berlin on Friday") == ["date"]
    assert kinds("Berlin in April") == ["date"]
    assert kinds("Berlin, may I ask? May is fine") == []  # the modal verb is not a month
    assert kinds("Berlin, Dana Castellano says hi") == ["name"]
    assert kinds("Berlin with Northwind Labs") == ["name"]
    assert kinds("Berlin, Dana's plan") == []  # names already in the template are fine
    assert kinds("Berlin at 9:00 on Friday with Hugo") == ["date", "name"]
    assert kinds("Berlin, then Seattle") == ["name"]  # an added city
    assert kinds("Berlin, then Austin, Sydney and Portland") == ["name"]
    assert kinds("Berlin, Berlin, Berlin") == []  # the template's own city is fine
    assert {c for c, _ in pools.CITIES} & llm_render._COMMON_WORD_NAMES == set()
    alt = (Anchor((("fully vegan",), ("Current diet: vegan",)), ("fully vegan",)),)
    assert check_body("Current diet: vegan", "I'm fully vegan", alt) == []


def test_clean_body_normalises_whitespace_only():
    raw = "Hi\xa0there  and\tyou \r\n\r\n\r\n\r\n next   line \n"
    assert clean_body(raw) == "Hi there and you\n\nnext line"


def test_requests_depend_on_the_revision_alone():
    anchors = (Anchor((("Berlin",),), ("Berlin",)),)
    a = render_request("messages", "Now in Berlin!", anchors)
    assert a == render_request("messages", "Now in Berlin!", anchors)
    assert a != render_request("notes", "Now in Berlin!", anchors)
    prompt = a["messages"][1]["content"]
    assert "<anchor>Berlin</anchor>" in prompt
    assert "<original>\nNow in Berlin!\n</original>" in prompt
    assert RETRY_MARK not in prompt and a["purpose"] == "render"
    assert a["schema"]["required"] == ["body"] and a["max_tokens"] >= 256
    retry = render_request("messages", "Now in Berlin!", anchors, [("number", "15")])
    assert RETRY_MARK in retry["messages"][1]["content"]
    assert "15" in retry["messages"][1]["content"]
    assert "(none)" in render_request("email", "Hello.", ())["messages"][1]["content"]


def test_render_is_faithful_and_leaves_gold_alone(worlds):
    world = worlds["src"]
    before = [e.payload["text"] for e in world.events]
    rendered, rep = render_world_llm(world, FakeLLM("fake/renderer"))
    assert [e.payload["text"] for e in world.events] == before  # input untouched
    assert rendered is not world and rendered.events is not world.events
    assert "render" not in world.params
    n_rewritten = 0
    for old, new in zip(world.events, rendered.events, strict=True):
        assert replace(old, payload={}) == replace(new, payload={})
        # only the text differs: body, sent, from_addr, conversation_with, ... are template values
        assert {k: v for k, v in old.payload.items() if k != "text"} == {
            k: v for k, v in new.payload.items() if k != "text"
        }
        if old.source in RENDER_SOURCES and old.op != "delete" and not _is_digest(old):
            n_rewritten += 1
            assert new.payload["text"] != old.payload["text"]
            assert _body(new, old).endswith(str(old.payload["body"]))  # the fake only prefixes
        else:
            assert new.payload["text"] == old.payload["text"]  # other sources, digests, deletes
    digests = [e for e in world.events if _is_digest(e)]
    total = sum(len(_events(world, s)) for s in RENDER_SOURCES)
    assert digests and n_rewritten == total - len(digests)
    assert rep.fidelity_rate == 1.0 and rep.fidelity_checked > 1000
    assert check_world(rendered).rate == 1.0
    for source, c in rep.sources.items():
        skipped = len(digests) if source == "email" else 0
        assert c.events == len(_events(world, source))
        assert c.skipped_background == skipped and c.rendered == c.events - skipped
        assert c.kept_template == 0 and c.retried == 0
    # gold is recomputed from payload fields the renderer does not touch
    log0, log1 = SourceLog(world.events), SourceLog(rendered.events)
    for t in (world.fork - timedelta(minutes=1), world.fork + timedelta(weeks=4)):
        for slot in world.slots:
            g0, g1 = gold_at(world, log0, slot, t), gold_at(rendered, log1, slot, t)
            assert g0.to_json() == g1.to_json()
    # the manifest hash identifies the rendered world
    rec0, rec1 = world_record(world), world_record(rendered)
    assert rec0["events_sha256"] != rec1["events_sha256"] and "render" not in rec0
    assert rec1["render"] == {
        "model": "fake/renderer",
        "sources": list(RENDER_SOURCES),
        "skipped_background": len(digests),
        "kept_template": 0,
    }
    replay, _ = render_world_llm(world, FakeLLM("fake/renderer"))
    assert world_record(replay) == rec1


def test_request_accounting_dedupe_and_sources(worlds):
    world = worlds["natural"]
    fake = FakeLLM("fake/renderer")
    _, rep = render_world_llm(world, fake)
    n_events = sum(c.events for c in rep.sources.values())
    assert rep.requests == fake.usage["calls"] == rep.usage["calls"]
    assert rep.requests < n_events  # repeated small talk is one request
    assert rep.to_json()["sources"]["messages"]["events"] == len(_events(world, "messages"))
    _, pooled = render_world_llm(world, FakeLLM("fake/renderer"), workers=4)
    assert pooled.to_json()["sources"] == rep.to_json()["sources"]
    only, notes_only = render_world_llm(world, FakeLLM(), sources=("notes",))
    assert set(notes_only.sources) == {"notes"}
    assert {e.source for e, o in zip(only.events, world.events, strict=True) if e != o} == {"notes"}


def test_retry_recovers_then_template_is_kept(worlds):
    world = worlds["src"]
    anchored = set(event_anchors(world))
    assert len(anchored) > 5
    # a renderer that drops an anchor once: every anchored event is rendered by its retry
    _, rep = render_world_llm(world, FakeLLM("fake/r", "drop_anchor", fix_on_retry=True))
    assert sum(c.rendered_retry for c in rep.sources.values()) == len(anchored)
    assert sum(c.kept_template for c in rep.sources.values()) == 0
    assert sum(c.first_failures.get("anchor", 0) for c in rep.sources.values()) == len(anchored)
    assert rep.fidelity_rate == 1.0
    # one that always does: the template body of those events is kept, the rest is rendered
    kept, rep = render_world_llm(world, FakeLLM("fake/r", "drop_anchor"))
    assert sum(c.kept_template for c in rep.sources.values()) == len(anchored)
    assert sum(c.retried for c in rep.sources.values()) == len(anchored)
    for old, new in zip(world.events, kept.events, strict=True):
        untouched = old.event_id in anchored or old.source not in RENDER_SOURCES
        untouched = untouched or is_background(old)
        assert (new.payload["text"] == old.payload["text"]) == untouched
    assert rep.fidelity_rate == 1.0
    assert kept.params["render"]["kept_template"] == len(anchored)


@pytest.mark.parametrize(
    "mode", ["add_number", "add_date", "add_name", "add_city", "bad_json", "cut_off"]
)
def test_faulty_renderers_fall_back_to_the_template(worlds, mode):
    world = worlds["natural"]
    kept, rep = render_world_llm(world, FakeLLM("fake/r", mode))
    attempted = sum(c.events - c.skipped_background for c in rep.sources.values())
    # a template body that already holds the "invented" content passes, so not quite all
    assert sum(c.kept_template for c in rep.sources.values()) > 0.9 * attempted
    reason = {"bad_json": "output", "cut_off": "output", "add_city": "name"}.get(
        mode, mode.removeprefix("add_")
    )
    assert sum(c.first_failures.get(reason, 0) for c in rep.sources.values()) > 0.9 * attempted
    assert rep.fidelity_rate == 1.0
    _, fixed = render_world_llm(world, FakeLLM("fake/r", mode, fix_on_retry=True))
    assert sum(c.kept_template for c in fixed.sources.values()) == 0


def test_digest_emails_are_skipped_and_counted(worlds):
    world = worlds["src"]
    digests = [e for e in world.events if _is_digest(e)]
    assert digests
    # the generator's digest markers agree with the one the renderer uses
    assert {e.event_id for e in digests} == {e.event_id for e in world.events if is_background(e)}
    assert {e.payload["cluster"] for e in digests} == {
        f"thread:digest-{t}" for t in pools.NEWSLETTER_TOPICS
    }
    assert all(" Digest #" in e.payload["subject"] for e in digests)
    fake = FakeLLM("fake/r")
    rendered, rep = render_world_llm(world, fake)
    by_id = {e.event_id: e for e in rendered.events}
    assert all(by_id[e.event_id] == e for e in digests)  # template body and text kept
    assert rep.sources["email"].skipped_background == len(digests)
    assert rep.sources["messages"].skipped_background == 0
    assert rep.sources["notes"].skipped_background == 0
    assert rep.to_json()["sources"]["email"]["skipped_background"] == len(digests)
    # nothing was requested for them: one request per distinct non-digest (source, body, anchors)
    anchors = event_anchors(world)
    distinct = {
        (e.source, str(e.payload["body"]), anchors.get(e.event_id, ()))
        for e in world.events
        if e.source in RENDER_SOURCES and e.op != "delete" and not _is_digest(e)
    }
    assert rep.requests == fake.usage["calls"] == len(distinct)


def test_delete_events_are_not_rendered(worlds):
    world = worlds["natural"]
    first = _events(world, "messages")[0]
    events = [replace(e, op="delete") if e is first else e for e in world.events]
    fake = FakeLLM("fake/r")
    rendered, rep = render_world_llm(replace(world, events=events), fake)
    assert rep.sources["messages"].events == len(_events(world, "messages")) - 1
    (new,) = [e for e in rendered.events if e.event_id == first.event_id]
    assert new.op == "delete" and new.payload["text"] == first.payload["text"]


@pytest.mark.parametrize("branch", ["natural", "src", "def", "orphan", "conflict"])
def test_worst_case_renderer_keeps_fidelity_by_construction(worlds, branch):
    # a renderer that writes nothing but the anchors still leaves every gold fact locatable
    rendered, rep = render_world_llm(worlds[branch], FakeLLM("fake/r", "anchors_only"))
    assert rep.fidelity_rate == 1.0 == check_world(rendered).rate
    assert sum(c.kept_template for c in rep.sources.values()) == 0


def test_fidelity_error_when_an_anchor_is_lost(worlds, monkeypatch):
    monkeypatch.setattr(llm_render, "check_body", lambda body, template, anchors: [])
    with pytest.raises(RenderFidelityError):
        render_world_llm(worlds["src"], FakeLLM("fake/r", "drop_anchor"))


def test_branches_share_rendered_natural_events(worlds):
    def natural_requests(world):
        anchors = event_anchors(world)
        return {
            (e.object_id, e.ts): render_request(
                e.source, str(e.payload["body"]), anchors.get(e.event_id, ())
            )
            for e in world.events
            if e.tag == "natural"
            and e.source in RENDER_SOURCES
            and e.op != "delete"
            and not is_background(e)
        }

    def natural_texts(world):
        return {
            (e.object_id, e.ts): e.payload["text"]
            for e in world.events
            if e.tag == "natural" and e.source in RENDER_SOURCES
        }

    base = worlds["natural"]
    base_rendered, _ = render_world_llm(base, FakeLLM("fake/r"))
    for branch in BRANCHES:
        if branch == "natural":
            continue
        # the request of a shared natural event is the same in every branch ...
        assert natural_requests(worlds[branch]) == natural_requests(base)
        # ... so its rendered text is byte for byte the same
        other, _ = render_world_llm(worlds[branch], FakeLLM("fake/r"))
        assert natural_texts(other) == natural_texts(base_rendered)


def test_e1_driver_renders_each_branch(tmp_path):
    params = {**(world_params(3) or {}), **SMALL}
    summary = e1.run_world(
        5,
        tmp_path,
        branches=("src", "def"),
        baselines=False,
        writer=FakeLLM(),
        cells_cfg=[e1.CellConfig(1, 1, "source")],
        world_params=params,
        renderer=FakeLLM("fake/renderer"),
    )
    assert len(summary["worlds"]) == 2
    for branch, rec in zip(("src", "def"), summary["worlds"], strict=True):
        template = generate_branch(5, branch, params)
        rendered, _ = render_world_llm(template, FakeLLM("fake/renderer"))
        assert rec == world_record(rendered) and rec["render"]["model"] == "fake/renderer"
        assert rec["events_sha256"] != world_record(template)["events_sha256"]
        report = summary["branches"][branch]["render"]
        assert report["model"] == "fake/renderer" and report["fidelity"]["rate"] == 1.0
        assert set(report["sources"]) == set(RENDER_SOURCES)
        assert summary["branches"][branch]["hash_ok"]


def test_e1_render_flag_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "client", lambda model, run_dir=None, **kw: FakeLLM(model))
    out = tmp_path / "run"
    out.mkdir()
    run_dir = e1.run(
        [5],
        out=out,
        cells_cfg=[e1.CellConfig(1, 1, "source")],
        baselines=False,
        branches=("src",),
        world_params={**(world_params(3) or {}), **SMALL},
        render_model="fake/renderer",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["params"]["render_model"] == "fake/renderer"
    assert manifest["models"]["renderer"]["model"] == "fake/renderer"
    assert manifest["worlds"][0]["render"]["model"] == "fake/renderer"
    (row,) = read_jsonl(run_dir / "world_summaries.jsonl")
    assert row["branches"]["src"]["render"]["sources"]["email"]["events"] > 0
    assert verify_manifest(run_dir) == []


def test_render_warm_cli_writes_reports_and_manifest(tmp_path, monkeypatch):
    import runpy
    import sys

    from memshift.experiments import common
    from tests.fakes import FakeLLM

    fake = FakeLLM()
    monkeypatch.setattr(common, "client", lambda model, run_dir, **kw: fake)
    monkeypatch.setattr(common, "new_run_dir", lambda name: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm_render", "--model", "m", "--seeds", "11", "--branches", "src", "--months", "3"],
    )
    runpy.run_module("memshift.gen.llm_render", run_name="__main__")
    rows = read_jsonl(tmp_path / "render_reports.jsonl")
    assert [(r["seed"], r["branch"]) for r in rows] == [(11, "src")]
    assert rows[0]["fidelity"]["rate"] == 1.0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["experiment"] == "L-render-warm" and len(manifest["worlds"]) == 1
