"""LLM-rendered worlds (prereg v2 section 5, block L): free-text bodies written by a model.

The template renderer (gen/render.py) produces every `payload["text"]`. Here the bodies of email,
chat-message and note events are rewritten by a model as natural text; headers and the structured
sources (contacts, calendar, tasks, settings) are untouched. Only `payload["text"]` changes: the
payload fields that gold, the kappa evaluators and the shift scripts read (body, sent, from_addr,
conversation_with, ...) keep their template values, so gold is identical to the template world.
The text is reassembled with render.py's header format around the new body. Delete events are not
rendered, and neither is background traffic that carries no gold fact: the newsletter digests
(prereg decision, block L). The generator marks them itself: `Builder._newsletters` (gen/world.py)
is the only caller of `Builder.email(..., thread=...)`, with `thread=f"thread:digest-{topic}"`,
which becomes `payload["cluster"]` (the sender is `digest@{topic}news.example`). Their template
body and text are kept and they are counted per source as `skipped_background`.

Requests are a function of the object revision alone (source, template body, anchors; no event
id, revision number, timestamp or branch), sent at temperature 0 through the response cache, so
the natural events shared by the branches of a world come back byte for byte identical. Identical
requests (repeated small talk) are sent once.

Anchors. Every gold fact whose surface form (gen/fidelity.py `surface_forms`) the template body
carries must survive verbatim, because the fidelity checker looks for it in the text of the
dependency objects. An anchor is a set of alternatives, satisfied by any one of them; the
alternative present in the template body is the one shown to the model. Derived per fact class of
gen/gold.py, from the object alone (no time sweep):
  - script-level user attributes (employer, manager, home city, diet, commute): every object
    listed in the `deps` of a timeline point is a positive dependency at some time; the anchor is
    the surface form of that point's value if the object's template body contains it (a revision
    that is listed but carries an older value gets no anchor for the newer one). Revision
    intervals are ignored on purpose: they depend on later timeline points, so a shared natural
    event would get different anchors, hence a different request, in different branches. The time
    zone depends on the calendar settings object only.
  - contact attributes, recurring series, appointments, task due dates: dependencies are contact
    cards, calendar objects and tasks, which are never rewritten; announcement emails and
    messages about them are not dependencies and carry no anchor beyond the guards below.
  - definition items (kappa): close friends and VIP senders depend on chat messages and emails
    through their headers (`conversation_with`, `from_addr`); the member's full name is an anchor
    when the body contains it (the templates put it in headers only, so this adds nothing there).
    Working hours v1 depends on the "Working hours" note: its sentence
    `My working hours: HH:MM-HH:MM`, the pattern the evaluator reads from `payload["body"]` and
    a D=1 quote of the note needs, is an anchor. The other definitions read calendar and tasks.
  - unresolvable conflicts (E7): the two contradicting values are anchors of the objects that
    announce them.

Guards on each rendered body, all mechanical: (1) the reply is a JSON object with a non-empty
"body" and was not cut off; (2) every anchor is present (exact, case-sensitive containment after
horizontal whitespace is collapsed, the strictest reading of fidelity.check_world, which is what
makes the fidelity rate 1.0 by construction; the prereg text says "verbatim"); (3) no number
(digit run, compared by value so "03" equals "3"), month name or weekday name that the template
body lacks; (4) no person name, company or city from gen/pools.py that the template body lacks
(a few person names that are also common English words are ignored; no city name collides with
one). A body that fails is requested once more with the problems listed; if it fails again the
template body is kept. Streets are constrained by the prompt only. Bodies outside +-50% of the
template's word count are counted, not rejected.

`render_world_llm` runs `fidelity.check_world` on the rendered world and raises if the rate is
below 1.0 (or, for a template world that itself is below 1.0, below the template's rate). A failed
model call raises; nothing is kept silently.
"""

from __future__ import annotations

import re
from calendar import month_name
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from memshift.gen import pools
from memshift.gen.fidelity import check_world, surface_forms
from memshift.gen.kappa import _WH_RE as WORKING_HOURS_RE
from memshift.gen.model import Event, Slot, World
from memshift.gen.render import render_text
from memshift.llm import LLMResult
from memshift.textutil import approx_tokens

RENDER_SOURCES = ("email", "messages", "notes")
DIGEST_CLUSTER_PREFIX = "thread:digest-"  # gen/world.py Builder._newsletters
RETRY_MARK = "Your previous rewrite was rejected"
MAX_NEW_TOKENS = 16384
LENGTH_BAND = (0.5, 1.5)

RENDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"body": {"type": "string"}},
    "required": ["body"],
    "additionalProperties": False,
}

SYSTEM = (
    "You rewrite short texts from a person's email, chat history and notes so that they read as "
    "natural human writing, keeping exactly what they say."
)

KIND = {
    "email": "an email body",
    "messages": "a chat message",
    "notes": "a personal note",
}

MEDIUM = {"email": "email", "messages": "chat", "notes": "note"}

RULES = (
    "- Keep the language (English) and roughly the same length (within 50% of the original).\n"
    "- Do not add any facts, names, dates, times, numbers, places or companies.\n"
    "- Keep every anchor string below exactly as written, character for character."
)

# Pool names that are also ordinary English words; a capitalised occurrence is not evidence of an
# invented person.
_COMMON_WORD_NAMES = frozenset({"Basil", "Bishop", "Wren", "Young"})


def is_background(e: Event) -> bool:
    """Newsletter digests: generated background volume, kept as template text."""
    return e.source == "email" and str(e.payload.get("cluster", "")).startswith(
        DIGEST_CLUSTER_PREFIX
    )


@dataclass(frozen=True)
class Anchor:
    """A gold surface form that must survive rewriting: any one alternative (a conjunction of
    strings that must all occur) satisfies it; `shown` is the alternative listed to the model."""

    alternatives: tuple[tuple[str, ...], ...]
    shown: tuple[str, ...]

    def satisfied(self, body: str) -> bool:
        return any(all(part in body for part in conj) for conj in self.alternatives)


class ChatClient(Protocol):
    model: str
    usage: dict[str, int]

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult: ...

    def map(self, requests: Sequence[dict[str, Any]]) -> list[LLMResult]: ...


class RenderFidelityError(RuntimeError):
    """The rendered world lost a gold fact that the template world located."""


# ------------------------------------------------------------------------------------- anchors
def _timeline_facts(world: World) -> dict[str, list[tuple[Slot, Any]]]:
    """object id -> (slot, value) facts that list the object as a dependency."""
    slots = {s.slot_id: s for s in world.slots}
    out: dict[str, list[tuple[Slot, Any]]] = {}
    for slot_id, points in world.timelines.items():
        for p in points:
            for oid in p.deps:
                out.setdefault(oid, []).append((slots[slot_id], p.value))
    for slot_id, conflict in world.params.get("conflicts", {}).items():
        for oid in conflict["objects"]:
            for value in conflict["values"]:
                out.setdefault(oid, []).append((slots[slot_id], value))
    return out


def _carried(slot: Slot, value: Any, body: str) -> Anchor | None:
    forms = surface_forms(slot, value)
    present = [conj for conj in forms if all(part in body for part in conj)]
    if not present:
        return None
    return Anchor(tuple(tuple(c) for c in forms), tuple(present[0]))


def event_anchors(
    world: World, sources: Sequence[str] = RENDER_SOURCES
) -> dict[str, tuple[Anchor, ...]]:
    """Anchors per event id (only events with at least one), sorted by their shown strings so the
    request of a shared natural event does not depend on the branch's timeline order."""
    facts = _timeline_facts(world)
    slots = {s.slot_id: s for s in world.slots}
    by_email = {p.email: p.name for p in world.people.values()}
    out: dict[str, tuple[Anchor, ...]] = {}
    for e in world.events:
        if e.source not in sources or e.op == "delete":
            continue
        body = str(e.payload["body"])
        candidates = list(facts.get(e.object_id, ()))
        if e.source == "messages" and "k.close_friend" in slots:
            candidates.append((slots["k.close_friend"], e.payload["conversation_with"]))
        if e.source == "email" and "k.vip_sender" in slots and e.payload["from_addr"] in by_email:
            candidates.append((slots["k.vip_sender"], by_email[e.payload["from_addr"]]))
        found = {a for slot, value in candidates if (a := _carried(slot, value, body)) is not None}
        sentence = (
            WORKING_HOURS_RE.search(body) if e.payload.get("title") == "Working hours" else None
        )
        if e.source == "notes" and sentence:
            found.add(Anchor(((sentence.group(0),),), (sentence.group(0),)))
        if found:
            out[e.event_id] = tuple(sorted(found, key=lambda a: (a.shown, a.alternatives)))
    return out


# --------------------------------------------------------------------------------------- guards
_NUMBER_RE = re.compile(r"\d+")
_CAPITALISED_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Za-z]*")
_DATE_WORDS = frozenset([*(m for m in month_name if m and m != "May"), *pools.WEEKDAYS])
_PERSON_NAMES = frozenset({*pools.FIRST_NAMES, *pools.LAST_NAMES}) - _COMMON_WORD_NAMES
_CITY_NAMES = frozenset(city for city, _ in pools.CITIES)


def _numbers(s: str) -> set[str]:
    """Digit runs compared by value (leading zeros dropped; no int(), which caps at 4300 digits)."""
    return {x.lstrip("0") or "0" for x in _NUMBER_RE.findall(s)}


def _dates(s: str) -> set[str]:
    return set(_CAPITALISED_RE.findall(s)) & _DATE_WORDS


def _names(s: str) -> set[str]:
    """Person names and cities (capitalised words) and companies (full names)."""
    words = set(_CAPITALISED_RE.findall(s))
    return (words & (_PERSON_NAMES | _CITY_NAMES)) | {o for o in pools.ORGS if o in s}


def clean_body(s: str) -> str:
    """Normalise whitespace only: no-break spaces, runs of spaces and tabs, blank-line runs."""
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


Issue = tuple[str, str]  # (kind, detail); kinds: output, anchor, number, date, name


def check_body(body: str, template: str, anchors: Sequence[Anchor]) -> list[Issue]:
    """Mechanical guards of a rendered body against its template body (empty list = accepted)."""
    issues: list[Issue] = [("anchor", ", ".join(a.shown)) for a in anchors if not a.satisfied(body)]
    if new := sorted(_numbers(body) - _numbers(template), key=lambda n: (len(n), n)):
        issues.append(("number", ", ".join(new)))
    if new_dates := sorted(_dates(body) - _dates(template)):
        issues.append(("date", ", ".join(new_dates)))
    if new_names := sorted(_names(body) - _names(template)):
        issues.append(("name", ", ".join(new_names)))
    return issues


_ISSUE_TEXT = {
    "output": 'The reply was not a JSON object with a non-empty "body" (or it was cut off).',
    "anchor": "An anchor string is missing; keep it verbatim: {}",
    "number": "The rewrite adds numbers that are not in the original: {}",
    "date": "The rewrite adds dates or weekdays that are not in the original: {}",
    "name": "The rewrite adds names, companies or cities that are not in the original: {}",
}


# ------------------------------------------------------------------------------------- requests
def render_prompt(
    source: str, body: str, anchors: Sequence[Anchor], issues: Sequence[Issue] = ()
) -> str:
    shown = [part for a in anchors for part in a.shown]
    listed = "\n".join(f"<anchor>{s}</anchor>" for s in shown) if shown else "(none)"
    lines = [
        f"Rewrite the text below ({KIND[source]}) as natural text a person would write in that "
        f"medium ({MEDIUM[source]}).",
        "",
        "Rules:",
        RULES,
        "",
        "Anchor strings:",
        listed,
        "",
        "Original text:",
        f"<original>\n{body}\n</original>",
        "",
    ]
    if issues:
        lines += [f"{RETRY_MARK}:"]
        lines += [f"- {_ISSUE_TEXT[k].format(d)}" for k, d in issues]
        lines += ["Rewrite the original text again and fix these problems.", ""]
    lines.append('Reply with JSON of the form {"body": "..."} containing only the rewritten text.')
    return "\n".join(lines)


def render_request(
    source: str, body: str, anchors: Sequence[Anchor], issues: Sequence[Issue] = ()
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": render_prompt(source, body, anchors, issues)},
        ],
        "max_tokens": min(MAX_NEW_TOKENS, 2 * approx_tokens(body) + 256),
        "schema": RENDER_SCHEMA,
        "purpose": "render",
    }


def parse_render(res: LLMResult) -> str | None:
    if res.finish_reason == "length":
        return None
    data = res.json()
    body = data.get("body") if isinstance(data, dict) else None
    if not isinstance(body, str):
        return None
    return clean_body(body) or None


# --------------------------------------------------------------------------------------- report
@dataclass
class SourceCounts:
    events: int = 0  # non-delete events of the source, including skipped background
    skipped_background: int = 0
    rendered_first: int = 0
    rendered_retry: int = 0
    kept_template: int = 0
    anchored_events: int = 0
    anchors: int = 0
    outside_length_band: int = 0  # rendered bodies outside +-50% of the template's word count
    first_failures: dict[str, int] = field(default_factory=dict)  # events per issue kind

    @property
    def rendered(self) -> int:
        return self.rendered_first + self.rendered_retry

    @property
    def retried(self) -> int:
        """Events whose first attempt failed a guard (a template is kept only after the retry)."""
        return self.rendered_retry + self.kept_template

    def to_json(self) -> dict[str, Any]:
        attempted = self.events - self.skipped_background
        return {
            "events": self.events,
            "skipped_background": self.skipped_background,
            "rendered": self.rendered,
            "retried": self.retried,
            "kept_template": self.kept_template,
            "kept_template_share": self.kept_template / attempted if attempted else 0.0,
            "anchored_events": self.anchored_events,
            "anchors": self.anchors,
            "outside_length_band": self.outside_length_band,
            "first_failures": dict(sorted(self.first_failures.items())),
        }


@dataclass
class RenderReport:
    model: str
    sources: dict[str, SourceCounts]
    requests: int  # distinct requests sent (first attempts plus retries)
    usage: dict[str, int]  # renderer client usage during this call
    fidelity_checked: int
    fidelity_rate: float
    fidelity_rate_template: float

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "sources": {s: c.to_json() for s, c in self.sources.items()},
            "requests": self.requests,
            "usage": self.usage,
            "fidelity": {
                "checked": self.fidelity_checked,
                "rate": self.fidelity_rate,
                "rate_template": self.fidelity_rate_template,
            },
        }


# ------------------------------------------------------------------------------------ rendering
@dataclass
class _Job:
    source: str
    template: str
    anchors: tuple[Anchor, ...]
    first_issues: list[Issue] = field(default_factory=list)
    body: str | None = None
    retried: bool = False


def _run(llm: ChatClient, requests: list[dict[str, Any]], workers: int | None) -> list[LLMResult]:
    if workers is None:
        return llm.map(requests)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(lambda r: llm.complete(**r), requests))


def _judge(job: _Job, res: LLMResult) -> tuple[str | None, list[Issue]]:
    body = parse_render(res)
    if body is None:
        return None, [("output", "")]
    issues = check_body(body, job.template, job.anchors)
    return (None if issues else body), issues


def _reassemble(e: Event, body: str) -> Event:
    text = render_text(e.source, {**e.payload, "body": body})
    return replace(e, payload={**e.payload, "text": text})


def render_world_llm(
    world: World,
    llm: ChatClient,
    sources: Sequence[str] = RENDER_SOURCES,
    workers: int | None = None,
) -> tuple[World, RenderReport]:
    """Rewrite the free-text bodies of `sources` with `llm`; returns a new world (the input is not
    modified) and the report. `workers` bounds a thread pool over single completions; by default
    the client's own `map` runs the requests."""
    if unknown := set(sources) - set(RENDER_SOURCES):
        raise ValueError(f"only free-text sources can be rendered, not {sorted(unknown)}")
    usage0 = dict(llm.usage)
    template_fidelity = check_world(world)
    anchors = event_anchors(world, sources)
    counts = {s: SourceCounts() for s in sources}
    jobs: dict[tuple[str, str, tuple[Anchor, ...]], _Job] = {}
    eligible: list[tuple[int, tuple[str, str, tuple[Anchor, ...]]]] = []
    for i, e in enumerate(world.events):
        if e.source not in sources or e.op == "delete":
            continue
        if is_background(e):
            counts[e.source].events += 1
            counts[e.source].skipped_background += 1
            continue
        key = (e.source, str(e.payload["body"]), anchors.get(e.event_id, ()))
        jobs.setdefault(key, _Job(*key))
        eligible.append((i, key))

    todo = list(jobs.values())
    results = _run(llm, [render_request(j.source, j.template, j.anchors) for j in todo], workers)
    failed: list[_Job] = []
    for job, res in zip(todo, results, strict=True):
        job.body, job.first_issues = _judge(job, res)
        if job.body is None:
            failed.append(job)
    retry_results = _run(
        llm,
        [render_request(j.source, j.template, j.anchors, j.first_issues) for j in failed],
        workers,
    )
    for job, res in zip(failed, retry_results, strict=True):
        job.retried = True
        job.body, _ = _judge(job, res)

    events = list(world.events)
    for i, key in eligible:
        job, c = jobs[key], counts[key[0]]
        c.events += 1
        c.anchored_events += bool(job.anchors)
        c.anchors += len(job.anchors)
        for kind in {k for k, _ in job.first_issues}:
            c.first_failures[kind] = c.first_failures.get(kind, 0) + 1
        if job.body is None:
            c.kept_template += 1
            continue
        c.rendered_retry += job.retried
        c.rendered_first += not job.retried
        words, before = len(job.body.split()), len(job.template.split())
        if before and not LENGTH_BAND[0] <= words / before <= LENGTH_BAND[1]:
            c.outside_length_band += 1
        events[i] = _reassemble(events[i], job.body)

    meta = {
        "model": llm.model,
        "sources": list(sources),
        "skipped_background": sum(c.skipped_background for c in counts.values()),
        "kept_template": sum(c.kept_template for c in counts.values()),
    }
    rendered = replace(world, events=events, params={**world.params, "render": meta})
    fidelity = check_world(rendered)
    if fidelity.rate < min(1.0, template_fidelity.rate):
        raise RenderFidelityError(
            f"{world.world_id} {world.branch}: fidelity {fidelity.rate:.4f} after rendering "
            f"(template {template_fidelity.rate:.4f}); first failures {fidelity.failures[:3]}"
        )
    report = RenderReport(
        model=llm.model,
        sources=counts,
        requests=len(todo) + len(failed),
        usage={k: v - usage0.get(k, 0) for k, v in llm.usage.items()},
        fidelity_checked=fidelity.checked,
        fidelity_rate=fidelity.rate,
        fidelity_rate_template=template_fidelity.rate,
    )
    return rendered, report


if __name__ == "__main__":
    # Block L render warm: render every (seed, branch) world with the renderer while it is served,
    # so the E1 `--render` run later reads each rendering from the request cache.
    import argparse
    import json

    from memshift.experiments.common import client, new_run_dir, world_params
    from memshift.experiments.e1 import parse_seeds
    from memshift.gen.shifts import generate_branch
    from memshift.runinfo import world_record, write_env, write_manifest

    ap = argparse.ArgumentParser(description="render worlds with an LLM to warm the cache")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--branches", default="src,def,orphan")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--name", default="render_warm")
    a = ap.parse_args()
    run_dir = new_run_dir(a.name)
    llm = client(a.model, run_dir)
    worlds = []
    for seed in parse_seeds(a.seeds):
        for branch in a.branches.split(","):
            world = generate_branch(seed, branch, world_params(a.months))
            rendered, rep = render_world_llm(world, llm)
            worlds.append(world_record(rendered))
            with (run_dir / "render_reports.jsonl").open("a") as f:
                f.write(json.dumps({"seed": seed, "branch": branch, **rep.to_json()}) + "\n")
    write_env(run_dir, {"renderer": a.model})
    params = {"seeds": parse_seeds(a.seeds), "branches": a.branches.split(","), "months": a.months}
    write_manifest(run_dir, "L-render-warm", worlds, {"renderer": a.model}, params)
    print(run_dir)
