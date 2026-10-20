"""The round-3 blocks U, N, H, Q, G and X, and the Claude blocks Y and V. Nothing here calls a
model.

One function per block takes the new runs and the committed runs they pair with and returns every
quantity the block computes and, under `threshold_tests`, each comparison of an estimate with the
threshold of the design (`tests`: key, value, threshold and `meets_threshold`, None when the value
is missing). A quantity without a threshold is under `reported`. Units are fractions (x 100 for
percent or pp).

    python -m memshift.analysis.round3_blocks BLOCK --runs RUN [RUN ...] \\
        [--compare RUN_OR_STAGE ...] [--response-cache SQLITE] --out FILE.json

  block  --runs (new)                              --compare (committed, paired with)
  U      the U runs: refresh and upsert cells,     e1_test_* and e1_synth_* (or the stages
         Qwen3.5-9B and the M2 families            stage2_e1, stage4_c_e1); m2_<family>_* for U5
  H      H-a (--defwriter claude-sonnet-5-5),      e1_test_*, e1_synth_*
         H-c (claude-opus-5-5), H-b (D0think 32k)
  G      G-a (--kappa-set heldout:4001..4003),     e1_test_*, e1_synth_* (the D=0 cells of G3 and
         G-b (one run set per program set)         the anchors of G-b)
  Q      the --probe-unnamed-defs runs             e1_test_*, e1_synth_*
  N      the natural-branch runs                   the main E1 stage (runs/stage2_e1), one path
  X      the x_mem0 runs                           none
  Y      the --reader claude-sonnet-5-5 runs       stage2_e1, stage4_c_e1 (the stores it reads)
  V      the --program-user-words runs, and the    stage2_e1, stage4_c_e1 (the D=0 cells)
         G-b runs of the same writers and samples

A `--compare` stage directory expands to the source runs its manifest lists
(`controls_data.expand_base`, which checks each source's manifest hash). The H validity check
reads the response cache of the Claude writer when `--response-cache` names it (otherwise
`calls.jsonl` cannot tell a parse failure from an answer, see below).

Estimators (`ex_e1_classes`, `scoring.stats`). Update is ContractValid on the changed item-times
(gold value or status differs from t0, `e1.item_table`); definition items are the fact ids `k.*`,
other items every other id. Every contrast is a per-item paired difference over the item-times
present in every cell it uses, clustered by world: 2,000 bootstrap draws of the worlds, seed 996,
percentile 95% interval. `R_x` is a ratio of pooled means, (mean x - mean B0) / (mean B1 - mean
B0) over the same item-times, with its world-bootstrap interval; it is not evaluated when the
pooled denominator is below 5 pp. B x D is (B1 - B0 at D=1) - (B1 - B0 at D=0) per item.

Two-way bootstrap (block G). The item-time values x (per-item double differences) are summed into
S[w, c] and counted into N[w, c] per world w and cluster c (a held-out definition, or a program
set). The point estimate is the pooled mean sum(S) / sum(N). A draw resamples the worlds (n_w with
replacement, the draws of `boot_index(n_w, 2000, 996)`, i.e. those of the one-way bootstrap) and,
independently, the clusters (n_c with replacement, `boot_index(n_c, 2000, 997)`), and takes the
product sample: with multiplicities a_w and b_c, the draw is sum a_w b_c S[w, c] / sum a_w b_c
N[w, c]. The interval is the 2.5 and 97.5 percentiles of the draws (`ex_e1_classes.interval`).
Worlds and clusters without item-times are left out of the resampling, as `world_bootstrap` drops
empty worlds. With one cluster it is the one-way world bootstrap.

Pairing (U, H, G, Q, Y, V). Only complete runs (manifest.json present) are read. For each
(group, cell, world, branch) the newest run (manifest `created_utc`) is used, and a run superseded
that way must have the kept run's answers. The group of a new run is its variant in H, G, Y and V
(what changes requests: writer, reader, definition writer, kappa set, program writer, sample and
wording, thinking limit, unnamed probes), its family in U and Q (writer, reader, kappa set, world
length: the refresh and upsert runs of one writer are merged); the group of a committed run is
its family, so committed copies of one cell in two runs (e1_test and e1_synth share D0B0 to
D1B1_source) must agree. A stage directory lists the runs it merged; passing it avoids superseded
copies. Then, per new run: every pre-shift hash assertion exists and passed; on each (world,
branch) the probe keys each cell was asked equal those of the committed reference cell (D0B0, else
another), and the gold of every shared probe id equals the committed gold (the question text too,
except the definition items of an unnamed-probe run); and the anchor answers (status, value,
ContractValid) equal the committed ones, on the probes whose question text is the same on both
sides. The anchor cells are every cell a new run shares with the committed runs of its family, at
every checkpoint, except: a run with a definition-item writer (H-a, H-c) at k = 0 only (its
definition items differ after the shift in every cell); a thinking run (H-b) D0B1_source at every
checkpoint, other shared cells at k = 0; a program-set run (G-b, V) its D1s cells at k = 0; a
held-out-definition run (G-a) has no committed counterpart and no anchor; a run with a swapped-in
reader (Y) pairs with the committed runs whose stores it reads (its family with the writer as
reader), and its anchor is the reader's context: the context hash of every answer equals the
committed one (its answers differ by design), and no write or program request was sent
(`store_requests_sent`). A definition-only run (`--post-probes definition`: G-a, G-b, V, U5) is
compared on what it answers: the t0 probes and the definition items of the committed draw after
the fork. N: the natural branch's k = 0 answers against the committed S_def branch's (same events
before the fork, same probes at t0). X: the anchor `experiments.x_mem0` records for a resumed E5
prefix. Every mismatch is counted under `pairing.counts`; a count that is not 0 makes the run void
(`pairing.ok` false), and its quantities are reported as computed. A void run is not repaired here.

Worlds. Each block is computed on fixed worlds (`BLOCK_WORLDS`: U, Q, G-b, Y and V 1000-1019; U5,
H, G-a and X 1000-1009; N through its item-time coverage); a comparison computed on fewer worlds
lists the missing ones under `worlds_missing`.

Comparisons and their thresholds (each paragraph is copied into the result under `notes`):

  U1     R_refresh >= 0.8 for each of the four (arm, shift) combinations (D=1 and D_m, S_src and
         S_def, definition items); an R that is not evaluated (denominator < 5 pp) has no value.
  U3     pooled over the four (arm, shift) combinations: one ratio of the item-times of all four,
         against 0.8 (executability) and 0.5 (key schedule), both bounds inclusive.
  U4     the D=1 arm (D1B0_upsert against D1B0), all items, S_del (branch orphan): the point
         estimate and the lower bound above 0.
  U5     per M2 family U3's ratio over D=1 (the M2 runs have no D_m) and S_src, S_def, against the
         same bounds.
  H1     B x D with the authored D=1 cells (the main-study contrast); the D_m version is
         reported. Bounds: at most 10 pp or the interval includes 0; at least 20 pp with the
         interval above 0.
  H2     r_0 = ContractValid of the changed definition item-times the source-epoch key
         invalidated in D0B1_source after S_src (the `r` of Eq. (1), whose main-study value is
         16.2%); r_0 - 0.162 >= 20 pp (point estimate). The same quantity of Qwen3.5-9B on the
         same worlds, and the paired difference, are reported.
  H      validity: at most 10% of the writer's distinct write requests stop at the output limit
         (`finish_reason` "length" in calls.jsonl: max_tokens or the context window), at most 2%
         unreadable (`within_limits`); H1 and H2 are computed only within these limits.
         calls.jsonl records the finish reason, prompt and output tokens (thinking included) but
         not whether the reply parsed; the unreadable share is per distinct request from the
         response cache when given, else from the run summaries' `def_stats` (per cell write, so a
         request several cells send counts once per cell), and `basis` says which.
  H3     repair and mechanical repair, S_src and S_def, D1B1_source minus D0thinkB1_source on
         definition items, each >= 30 pp; computed only when at most 10% of the thinking writes
         are cut off (completion tokens at the run's limit). The share of unreadable thinking
         writes against 20% is reported beside it.
  G1-G3  lower bound of the two-way interval > 0. G1 and G2 need both arms (D=1 and D_m) and all
         three draws (18 definitions); G3 is computed on the program sets given (at least 2; the
         cut order can leave fewer than 18) and says how many. The share between writers is the
         eta squared of the per-set B x D (and the one-way random-effects share); the S_del
         interaction per writer pools its sets (world bootstrap).
  Q1     point estimate of (B effect at D=1, unnamed) - (B effect at D=1, named), paired per item,
         >= 5 pp. Q2 is computed at the authored D=1 (D_m reported), lower bound > 0.
  N1     `ex_round3_natural.paired_vs_natural`, part "shift", definition items, S_src and S_def,
         lower bound > 0; a comparison has no outcome (`coverage`) unless the natural answers
         cover every item-time.
  X1     per cell with a diagnosis and both effects: the arm that targets the dominant failure
         raises Update more than the other arm (point estimates); at least 3 such cells, and the
         target arm wins in at least 4 of them, or in all of them when fewer than 4 are compared.
         `analysis.x_mem0` computes the cells. In a void run (a pairing count not 0) X1's
         comparisons are not computed; the cells (m, r, a, the arms' effects) are reported.
  Y      validity: the reader's answers with status "error" (unreadable) and its calls with
         finish_reason "length" (cut off at the output limit) are counted separately over every
         answer it gave, json arm included (an answer both cut off and unreadable counts twice);
         at most 2% in all (`within_limits`). The context anchor voids a comparison when a
         context differs in a cell it uses (Y1-Y3: D0B0 to D1B1_source); an answer whose
         context differs in another cell (a D_m program that failed in the committed run and ran
         in this one, or the reverse) did not read the committed store and is left out of every
         quantity, and `pairing.detail.context_differs` lists it.
  Y1-Y3  B x D with the authored D=1 cells, on the reader's own item-times (those of the
         committed cells when the pairing holds). Y1 (S_def) and Y2 (S_del), the 42 other items:
         lower bound > 0. Y3: the paired per-item difference of the double differences, Sonnet
         reader minus Qwen3.5-9B reader, over the item-times both answered; |point estimate| <= 5
         pp after S_src and after S_def. The store/reader split is `ex_round3_store_reader`'s on
         the reader's answers: the stores are the committed ones, so its store part is the
         committed analysis's and the paired difference of B x D is a difference of reader parts.
  V1-V2  as G3: B x D at D_m on definition items, two-way bootstrap over worlds x the V program
         sets, lower bound > 0 (V1 S_def, V2 S_src), on the nine sets of the design (Qwen3.5-9B at
         temperature 0.7, Claude Sonnet 5.5 and Claude Opus 5.5, samples 0-2); `sets` says how many
         are given.
  V      reported per writer: V minus G-b on the sets of the same writer and sample index, in
         Update on the changed definition item-times of D1sB1_source (per branch and the three
         pooled) and in the agreement of the definition programs' t0 value with gold (as
         `e1_synth.program_stats`'s agree_t0), each paired per item, world bootstrap.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import controls_data as cd
from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_round2 as r2
from memshift.analysis import ex_round3_natural as natmod
from memshift.analysis import ex_round3_store_reader as srmod
from memshift.analysis import x_mem0 as xa
from memshift.experiments.common import MAIN_MODEL
from memshift.holdmem.writer import CUT_OFF
from memshift.llm import parse_json
from memshift.runinfo import code_state
from memshift.scoring.stats import item_diffs

Row = e1.Row
Vec = ex.Vec
Est = ex.Est
Cond = e1.Cond

STATUS = (
    "round-3 blocks: estimates with their world-clustered bootstrap intervals, and each comparison "
    "with the threshold of the design"
)
UNITS = "fractions (x 100 for percent or pp); 95% world-clustered bootstrap, 2,000 draws, seed 996"
DRAWS = ex.DRAWS
SEED = ex.SEED  # the world draws, those of `scoring.stats`
SEED_CLUSTERS = SEED + 1  # the definition or program-set draws of the two-way bootstrap
EPS = 1e-12
BRANCHES = ("src", "def", "orphan")  # S_src, S_def, S_del
SHIFTS = ("src", "def")
CLASSES = ("definition", "other", "all")
ANSWER_FIELDS = cd.ANSWER_FIELDS  # what an anchor must reproduce

# arm -> (B=0 cell, B=1 source-epoch cell); D_m is the writer-authored derivation (block C)
ARMS: dict[str, tuple[str, str]] = {
    "D0": ("holdmem_D0B0", "holdmem_D0B1_source"),
    "D1": ("holdmem_D1B0", "holdmem_D1B1_source"),
    "Dm": ("holdmem_D1sB0", "holdmem_D1sB1_source"),
}
ARM_CELLS = frozenset(c for pair in ARMS.values() for c in pair)
THINK_CELL = "holdmem_D0thinkB1_source"

# block U
U_KINDS = ("refresh", "upsert")
MIN_DENOMINATOR = 0.05
U1_MIN = 0.8
U3_COMBINATIONS = ("D1/src", "D1/def", "Dm/src", "Dm/def")  # "pooled over the four"
U5_COMBINATIONS = ("D1/src", "D1/def")  # the M2 runs have no D_m cell
# block H
MAX_AT_LIMIT, MAX_UNREADABLE = 0.10, 0.02  # H-a / H-c validity
QWEN_R0 = 0.162  # H2: Qwen3.5-9B's r_0 on changed definition items after S_src (main study)
H2_MARGIN = 0.20
SONNET_R0_BOUND = 0.70  # Sonnet's r_0 against 70% (reported)
H3_MIN = 0.30
MAX_THINK_CUT = 0.10  # H-b validity (writes cut off); the controls' D2 reads 20% unreadable
MAX_THINK_UNREADABLE = 0.20
# block G
G_DRAWS = 3  # three draws of six held-out definitions
G_SETS = 18
G3_MIN_SETS = 2
# block Q
Q1_MARGIN = 0.05
# block X
X_MIN_CELLS, X_MIN_FAVOUR = 3, 4
# blocks Y and V (the Claude reader and the program sets with user words)
Y3_MARGIN = 0.05  # |B x D (Sonnet reader) - B x D (Qwen3.5-9B reader)|, definition items, D=1
V_SETS = 9  # Qwen3.5-9B, Claude Sonnet 5.5 and Claude Opus 5.5, three samples each


def _seeds(lo: int, hi: int) -> tuple[str, ...]:
    return tuple(f"w{s}" for s in range(lo, hi + 1))


# the worlds each block is computed on
BLOCK_WORLDS: dict[str, tuple[str, ...]] = {
    "U": _seeds(1000, 1019),
    "U5": _seeds(1000, 1009),
    "H": _seeds(1000, 1009),
    "G-a": _seeds(1000, 1009),
    "G-b": _seeds(1000, 1019),
    "Q": _seeds(1000, 1019),
    "X": _seeds(1000, 1009),
    "Y": _seeds(1000, 1019),
    "V": _seeds(1000, 1019),
}

_SYSTEM = re.compile(r'"system": "([^"]*)"')
_STAMP = re.compile(r"(\d{8}T\d{6}Z)$")


# ===================================================================================== runs
def _rows(path: Path, keep: Callable[[str], bool] | None = None) -> list[Row]:
    """A JSONL file; with `keep`, only the rows whose (first) `system` field it accepts, without
    parsing the others."""
    if not path.exists():
        return []
    out: list[Row] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if keep is not None:
                m = _SYSTEM.search(line)
                if m is None or not keep(m.group(1)):
                    continue
            out.append(json.loads(line))
    return out


def _model(x: Any) -> str | None:
    """A manifest's model entry: a record with `model`, or the bare name of older manifests."""
    return str(x.get("model")) if isinstance(x, dict) else (str(x) if x else None)


def _k(row: Mapping[str, Any]) -> int:
    return int((row.get("tags") or {}).get("k", 0))


@dataclass
class Run:
    """One run directory: its manifest and the rows read from it."""

    path: Path
    manifest: dict[str, Any]
    probes: list[Row] = field(default_factory=list)
    answers: list[Row] = field(default_factory=list)
    items: list[Row] = field(default_factory=list)
    by_cell: dict[tuple[str, str, str], dict[str, Row]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a in self.answers:
            key = (a["system"], a["world_id"], a["branch"])
            self.by_cell.setdefault(key, {})[a["probe_id"]] = a

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.manifest.get("params") or {})

    @property
    def created(self) -> str:
        c = self.manifest.get("created_utc")
        if c:
            return str(c)
        m = _STAMP.search(self.path.name)
        return m.group(1) if m else ""

    @property
    def writer(self) -> str:
        return _model((self.manifest.get("models") or {}).get("writer")) or MAIN_MODEL

    @property
    def reader(self) -> str:
        return _model((self.manifest.get("models") or {}).get("reader")) or self.writer

    @property
    def world_params(self) -> dict[str, Any]:
        return dict(self.params.get("world_params") or {})

    @property
    def kappa_set(self) -> str | None:
        ks = self.world_params.get("kappa_set")
        return None if ks in (None, "", "fixed") else str(ks)

    @property
    def defwriter(self) -> str | None:
        d = self.params.get("defwriter")
        return _model(d) if d else None

    @property
    def think_max_tokens(self) -> int | None:
        tw = self.params.get("think_writer")
        return int(tw["max_tokens"]) if isinstance(tw, dict) and "max_tokens" in tw else None

    @property
    def program(self) -> tuple[str, float, int | None] | None:
        """The program writer when it is not the default (the writer, temperature 0, no sample)."""
        pw = self.params.get("program_writer")
        if not isinstance(pw, dict):
            return None
        model = str(pw.get("model") or self.writer)
        temp = float(pw.get("temperature") or 0.0)
        sample = pw.get("sample")
        if model == self.writer and not temp and sample is None:
            return None
        return model, temp, None if sample is None else int(sample)

    @property
    def user_words(self) -> bool:
        """Block V: the definition stated in the user's own words in the D_m program request."""
        pw = self.params.get("program_writer")
        return bool(isinstance(pw, dict) and pw.get("user_words"))

    @property
    def unnamed(self) -> bool:
        return bool(self.params.get("probe_unnamed_defs"))

    @property
    def definition_only(self) -> bool:
        """`--post-probes definition`: after the fork only the definition items of the
        discriminative draw are answered (every t0 probe is)."""
        return self.params.get("post_probes") == "definition"

    def keeps(self, fact: str, k: int) -> bool:
        """Whether the run answers a probe of the committed runs' draw."""
        return not self.definition_only or k == 0 or fact.startswith("k.")

    @property
    def cells(self) -> list[str]:
        return [str(c) for c in self.params.get("cells") or []]

    @property
    def family(self) -> str:
        """Writer, reader, definitions and world length: what committed runs pair on."""
        months = int(self.world_params.get("months", 12))
        return json.dumps([self.writer, self.reader, self.kappa_set, months])

    @property
    def store_family(self) -> str:
        """The family of the committed runs whose stores a swapped-in reader reads (block Y):
        the run's family with the writer as reader."""
        months = int(self.world_params.get("months", 12))
        return json.dumps([self.writer, self.writer, self.kappa_set, months])

    @property
    def variant(self) -> str:
        """Everything that changes a request: runs of one variant answer the same cells alike."""
        return json.dumps(
            {
                "family": self.family,
                "defwriter": self.defwriter,
                "program": self.program,
                "user_words": self.user_words,
                "think_max_tokens": self.think_max_tokens,
                "unnamed": self.unnamed,
                "render": self.params.get("render_model"),
            },
            sort_keys=True,
        )

    def asked(self, system: str, world: str, branch: str) -> set[tuple[str, int]]:
        return {
            (a["fact_id"], _k(a)) for a in self.by_cell.get((system, world, branch), {}).values()
        }


def read_runs(
    paths: Iterable[Path],
    systems: Callable[[str], bool] | None = None,
    items: Callable[[str], bool] | None = None,
) -> tuple[list[Run], list[str]]:
    """The complete runs among `paths` (manifest.json present) and the names of the others.
    `systems` keeps the answers of some cells only; `items` reads items.jsonl for some cells."""
    runs: list[Run] = []
    skipped: list[str] = []
    for p in paths:
        man = p / "manifest.json"
        if not man.exists():
            skipped.append(p.name)
            continue
        runs.append(
            Run(
                p,
                dict(json.loads(man.read_text())),
                _rows(p / "probes.jsonl"),
                _rows(p / "answers.jsonl", systems),
                _rows(p / "items.jsonl", items) if items is not None else [],
            )
        )
    return runs, skipped


def expand(paths: Sequence[Path]) -> list[Path]:
    """Run directories as given; a stage directory expands to its source runs."""
    return cd.expand_base(paths) if paths else []


# ===================================================================================== merge
Owner = tuple[str, str, str, str]  # (group, system, world, branch)


def _answer_diff(
    a: Mapping[str, Row],
    b: Mapping[str, Row],
    keep: Callable[[str], bool] = lambda pid: True,
    fields: Sequence[str] = ANSWER_FIELDS,
) -> dict[str, int]:
    """Answers of two copies of a cell compared on the probe ids `keep` accepts."""
    ids = [pid for pid in sorted(a.keys() | b.keys()) if keep(pid)]
    missing = sum(pid not in a or pid not in b for pid in ids)
    differ = sum(
        pid in a and pid in b and any(a[pid].get(f) != b[pid].get(f) for f in fields) for pid in ids
    )
    return {"compared": len(ids) - missing, "differ": differ, "missing": missing}


@dataclass
class Merged:
    """Runs merged per group (variant or family): one owner run per (group, cell, world,
    branch), the newest complete one."""

    runs: list[Run]
    group: Callable[[Run], str]
    owner: dict[Owner, Run] = field(default_factory=dict)
    superseded: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        cands: dict[Owner, list[Run]] = defaultdict(list)
        for run in self.runs:
            for system, w, b in run.by_cell:
                cands[(self.group(run), system, w, b)].append(run)
        for key, rs in sorted(cands.items()):
            rs = sorted(rs, key=lambda r: (r.created, r.name), reverse=True)
            self.owner[key] = rs[0]
            cell = key[1:]
            for r in rs[1:]:
                diff = _answer_diff(r.by_cell[cell], rs[0].by_cell[cell])
                self.superseded.append(
                    {"run": r.name, "kept": rs[0].name, "cell": list(cell), **diff}
                )

    def groups(self) -> list[str]:
        return sorted({g for g, *_ in self.owner})

    def runs_of(self, group: str) -> list[Run]:
        return [r for r in self.runs if self.group(r) == group]

    def answers(self, group: str, system: str, world: str, branch: str) -> dict[str, Row] | None:
        run = self.owner.get((group, system, world, branch))
        return None if run is None else run.by_cell[(system, world, branch)]

    def probes(self, group: str) -> dict[str, Row]:
        """Probe id -> probe, from the runs that own a cell of the group."""
        out: dict[str, Row] = {}
        owners = {id(r): r for (g, *_), r in self.owner.items() if g == group}
        for run in owners.values():
            for p in run.probes:
                out.setdefault(p["probe_id"], p)
        return out

    def data(self, group: str) -> dict[str, list[Row]]:
        """Probes, answers and items (`e1.load`'s shape) of the cells the group's owner runs
        hold."""
        owned = {(s, w, b): r for (g, s, w, b), r in self.owner.items() if g == group}
        runs = {id(r): r for r in owned.values()}.values()
        cells_wb = {(w, b) for _, w, b in owned}
        probes: dict[str, Row] = {}
        for run in runs:
            for p in run.probes:
                if (p["world_id"], p["branch"]) in cells_wb:
                    probes.setdefault(p["probe_id"], p)
        answers = [a for cell, run in owned.items() for a in run.by_cell[cell].values()]
        items = [
            it
            for run in runs
            for it in run.items
            if owned.get((it["system"], it["world_id"], it["branch"])) is run
        ]
        return {"probes": list(probes.values()), "answers": answers, "items": items}

    def table(self, group: str) -> ex.Table:
        """The E1 item table (`e1.item_table`) of the cells the group's owner runs hold."""
        return ex.Table(e1.item_table(self.data(group)))


def by_family(run: Run) -> str:
    return run.family


def by_variant(run: Run) -> str:
    return run.variant


def by_store_family(run: Run) -> str:
    return run.store_family


# =================================================================================== pairing
AnchorScope = Callable[[Run, str], str | None]  # "all", "k0", "context" or None (no anchor)
CONTEXT_FIELDS = ("context_sha256",)  # what a swapped-in reader's anchor must reproduce


def _scope_shared(run: Run, system: str) -> str | None:
    """U and Q: every cell a new run shares with the committed runs, at every checkpoint."""
    return "all"


def _scope_context(run: Run, system: str) -> str | None:
    """Y: every cell at every checkpoint, on the reader's context (its answers differ by design)."""
    return "context"


def _scope_h(run: Run, system: str) -> str | None:
    if run.defwriter:  # its definition items after the shift differ in every cell
        return "k0"
    if THINK_CELL in run.cells:  # H-b: D0B1_source is the anchor at every checkpoint
        return "all" if system == "holdmem_D0B1_source" else "k0"
    return "all"


def _scope_g(run: Run, system: str) -> str | None:
    if run.kappa_set:
        return None  # held-out definitions: no committed counterpart
    if run.program is not None and system.startswith("holdmem_D1s"):
        return "k0"  # the program set changes the D1s cells after the shift
    return "all"


def _hash(runs: Sequence[Run]) -> dict[str, int]:
    failed = missing = 0
    for run in runs:
        files = {f.name: f for f in run.path.glob("hash_check_*.json")}
        for w, b in sorted({(w, b) for _, w, b in run.by_cell}):
            f = files.get(f"hash_check_{w}_{b}.json")
            if f is None:
                missing += 1
            elif not json.loads(f.read_text()).get("ok"):
                failed += 1
    return {"hash_failed": failed, "hash_missing": missing}


def call_counts(run: Run) -> dict[str, int]:
    """The run's model calls (calls.jsonl) by purpose and whether the response cache served
    them: reported, without a threshold (U, Q and N record writes that are all cache hits)."""
    out: dict[str, int] = defaultdict(int)
    for c in _rows(run.path / "calls.jsonl"):
        out[f"{c.get('purpose')}/{'cached' if c.get('cached') else 'sent'}"] += 1
    return dict(sorted(out.items()))


def _reference(cmp: Merged, family: str, world: str, branch: str) -> set[tuple[str, int]] | None:
    """The probe keys the committed reference cell (D0B0, else the first cell) was asked."""
    systems = sorted(s for g, s, w, b in cmp.owner if (g, w, b) == (family, world, branch))
    if not systems:
        return None
    ref = "holdmem_D0B0" if "holdmem_D0B0" in systems else systems[0]
    answers = cmp.answers(family, ref, world, branch) or {}
    return {(a["fact_id"], _k(a)) for a in answers.values()}


def _pair_run(
    run: Run,
    cmp: Merged,
    ref_probes: Mapping[str, Row],
    scope: AnchorScope,
    rd: dict[str, int],
    family: Callable[[Run], str] = by_family,
    differing: list[tuple[str, str, str, str]] | None = None,
) -> None:
    """Probe keys, gold and anchors of one new run against the committed runs of its family.
    A definition-only run (`--post-probes definition`) is compared on the probes it answers: the
    committed draw restricted to the t0 probes and the definition items after the fork."""
    fam = family(run)
    for w, b in sorted({(w, b) for _, w, b in run.by_cell}):
        ref = _reference(cmp, fam, w, b)
        if ref is None:
            rd["no_committed_reference"] += 1
            continue
        ref = {(f, k) for f, k in ref if run.keeps(f, k)}
        for system in sorted({s for s, ww, bb in run.by_cell if (ww, bb) == (w, b)}):
            asked = run.asked(system, w, b)
            rd["probe_keys_missing"] += len(ref - asked)
            rd["probe_keys_extra"] += len(asked - ref)
    texts = {p["probe_id"]: p.get("question_text") for p in run.probes}
    for p in run.probes:
        q = ref_probes.get(p["probe_id"])
        if q is None:
            continue
        # a definition-only run records the other items after the fork as not asked
        fields: tuple[str, ...] = ("gold_value", "gold_status", "asked")
        if not run.keeps(str(p["fact_id"]), _k(p)):
            fields = fields[:2]
        if any(p.get(f) != q.get(f) for f in fields):
            rd["gold_differs"] += 1
        unnamed_def = run.unnamed and str(p["fact_id"]).startswith("k.")
        if p.get("question_text") != q.get("question_text") and not unnamed_def:
            rd["question_differs"] += 1

    def same_text(pid: str) -> bool:
        q = ref_probes.get(pid)
        return q is None or q.get("question_text") == texts.get(pid)

    anchored = 0
    for (system, w, b), mine in sorted(run.by_cell.items()):
        how = scope(run, system)
        theirs = cmp.answers(fam, system, w, b)
        if how is None or theirs is None:
            continue
        theirs = {pid: a for pid, a in theirs.items() if run.keeps(a["fact_id"], _k(a))}
        if how == "k0":
            mine = {pid: a for pid, a in mine.items() if _k(a) == 0}
            theirs = {pid: a for pid, a in theirs.items() if _k(a) == 0}
        if how == "context":
            diff = _answer_diff(mine, theirs, same_text, CONTEXT_FIELDS)
            rd["context_hash_differs"] += diff["differ"]
            rd["context_hash_missing"] += diff["missing"]
            if differing is not None:
                differing += [
                    (system, w, b, pid)
                    for pid in sorted(mine.keys() & theirs.keys())
                    if same_text(pid)
                    and mine[pid].get("context_sha256") != theirs[pid].get("context_sha256")
                ]
        else:
            diff = _answer_diff(mine, theirs, same_text)
            rd["anchor_answers_differ"] += diff["differ"]
            rd["anchor_answers_missing"] += diff["missing"]
        anchored += diff["compared"]
        rd["anchor_answers_compared"] += diff["compared"]
    if not anchored:
        rd["runs_without_anchor"] += 1


def pairing(
    new: Merged,
    cmp: Merged | None,
    skipped: Sequence[str],
    scope: AnchorScope | None,
    family: Callable[[Run], str] = by_family,
) -> dict[str, Any]:
    """The integrity checks of the module docstring, as counts (`ok` when every count is 0).
    `family` names the committed family a new run pairs with (Y: its stores' family)."""
    counts: dict[str, int] = defaultdict(int)
    detail: dict[str, Any] = {"incomplete_runs_skipped": list(skipped), "runs": {}}
    for m in (new, cmp):
        if m is None:
            continue
        for s in m.superseded:
            counts["superseded_answers_differ"] += s["differ"] + s["missing"]
    detail["superseded"] = [*new.superseded, *(cmp.superseded if cmp else [])]
    detail["calls"] = {run.name: call_counts(run) for run in new.runs}
    counts.update(_hash(new.runs))
    cmp_probes = {f: cmp.probes(f) for f in cmp.groups()} if cmp is not None else {}
    differing: list[tuple[str, str, str, str]] = []
    for run in new.runs:
        rd: dict[str, int] = defaultdict(int)
        paired = scope is not None and any(scope(run, s) is not None for s, _, _ in run.by_cell)
        if paired and scope is not None and cmp is not None:
            _pair_run(run, cmp, cmp_probes.get(family(run), {}), scope, rd, family, differing)
        detail["runs"][run.name] = dict(rd)
        for k, v in rd.items():
            if k != "anchor_answers_compared":
                counts[k] += v
    if differing:
        where: dict[str, int] = defaultdict(int)
        for system, w, b, _ in differing:
            where[f"{system}|{w}|{b}"] += 1
        detail["context_differs"] = {"by_cell": dict(sorted(where.items())), "answers": differing}
    return {
        "ok": not any(counts.values()),
        "counts": dict(sorted(counts.items())),
        "anchor_answers_compared": sum(
            r.get("anchor_answers_compared", 0) for r in detail["runs"].values()
        ),
        "detail": detail,
    }


# ================================================================================ estimators
def changed(cls: str) -> Cond:
    return lambda r: bool(r["changed"]) and ex.in_class(r, cls)


def _detected_changed(cls: str) -> Cond:
    return lambda r: bool(r["changed"]) and bool(r.get("detected")) and ex.in_class(r, cls)


def _repaired(cls: str) -> Cond:
    return lambda r: bool(r.get("repaired_now")) and ex.in_class(r, cls)


METRICS: dict[str, tuple[str, Callable[[str], Cond]]] = {
    "update": ("cv", changed),
    "repair": ("cv", _repaired),
    "repair_mech": ("repair_mech", _repaired),
}


def mvec(tab: ex.Table, system: str, branch: str, metric: str, cls: str) -> Vec:
    fld, cond = METRICS[metric]
    return tab.vec(system, branch, fld, cond(cls))


def _common(*vecs: Mapping[str, Mapping[str, float]]) -> dict[str, set[str]]:
    """Per world, the item keys present in every vector."""
    worlds = set.intersection(*({w for w, d in v.items() if d} for v in vecs)) if vecs else set()
    out = {w: set.intersection(*(set(v[w]) for v in vecs)) for w in sorted(worlds)}
    return {w: ks for w, ks in out.items() if ks}


def _restrict(v: Mapping[str, Mapping[str, float]], keys: Mapping[str, set[str]]) -> Vec:
    return {w: {k: v[w][k] for k in sorted(ks)} for w, ks in keys.items()}


def _n_items(keys: Mapping[str, Collection[str]]) -> int:
    return sum(len(v) for v in keys.values())


def nan_est() -> Est:
    return ex.interval(float("nan"), np.array([]), 0, 0)


def _present(e: Mapping[str, Any] | None) -> bool:
    return e is not None and e.get("n_worlds", 0) > 0 and not math.isnan(float(e["est"]))


def ratio_estimate(num: Vec, den: Vec, draws: int = DRAWS) -> Est:
    """Ratio of pooled paired means, sum(num) / sum(den) over the items in both, world-clustered
    bootstrap (the estimator of `controls.ratio_estimate`, vectorised; a draw whose denominator is
    0 is dropped)."""
    rows = []
    for w in sorted(w for w in num if w in den):
        ks = num[w].keys() & den[w].keys()
        if ks:
            rows.append([sum(num[w][k] for k in ks), sum(den[w][k] for k in ks), len(ks)])
    arr = np.array(rows, dtype=float).reshape(-1, 3)

    def stat(t: np.ndarray) -> Any:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(t[..., 1] != 0, t[..., 0] / t[..., 1], np.nan)

    return ex.world_estimate(arr, stat, int(arr[:, 2].sum()), draws)


def double_diff(u11: Vec, u10: Vec, u01: Vec, u00: Vec) -> Vec:
    """Per item (B1 - B0 at D=1) - (B1 - B0 at D=0), over the items present in all four."""
    keys = _common(u11, u10, u01, u00)
    return {
        w: {k: (u11[w][k] - u10[w][k]) - (u01[w][k] - u00[w][k]) for k in sorted(ks)}
        for w, ks in keys.items()
    }


def getter(tab: ex.Table, branch: str, cond: Cond) -> Callable[[str], Vec]:
    """system -> its ContractValid vector on one branch, rows that meet `cond`."""

    def get(system: str) -> Vec:
        return tab.vec(system, branch, "cv", cond)

    return get


def bxd(get: Callable[[str], Vec], arm: str = "D1") -> Est:
    """B x D on the items `get(system)` returns, with `arm` as the D=1 arm."""
    lo, hi = ARMS[arm]
    return ex.vec_mean(double_diff(get(hi), get(lo), get(ARMS["D0"][1]), get(ARMS["D0"][0])))


def _multiplicities(idx: np.ndarray, n: int) -> np.ndarray:
    """(draws, n) counts of each position in each draw of `boot_index`."""
    m = np.zeros((idx.shape[0], n), dtype=float)
    np.add.at(m, (np.arange(idx.shape[0])[:, None], idx), 1.0)
    return m


def two_way(
    s: np.ndarray,
    n: np.ndarray,
    draws: int = DRAWS,
    seed: int = SEED,
    seed_clusters: int = SEED_CLUSTERS,
) -> Est:
    """The two-way bootstrap of the module docstring: `s` and `n` are (worlds, clusters) sums and
    counts of the item values; worlds and clusters are resampled independently."""
    s, n = np.asarray(s, dtype=float), np.asarray(n, dtype=float)
    rows, cols = n.sum(axis=1) > 0, n.sum(axis=0) > 0
    s, n = s[rows][:, cols], n[rows][:, cols]
    nw, nc = s.shape
    if nw == 0 or nc == 0:
        return {**nan_est(), "n_clusters": 0}
    est = float(s.sum() / n.sum())
    a = _multiplicities(ex.boot_index(nw, draws, seed), nw)
    b = _multiplicities(ex.boot_index(nc, draws, seed_clusters), nc)
    num = np.einsum("dw,wc,dc->d", a, s, b)
    den = np.einsum("dw,wc,dc->d", a, n, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(den > 0, num / den, np.nan)
    return {**ex.interval(est, vals, nw, int(n.sum())), "n_clusters": int(nc)}


def cluster_arrays(parts: Mapping[str, Vec]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """(worlds, clusters, S, N) of per-cluster item vectors (world -> item -> value)."""
    worlds = sorted({w for v in parts.values() for w, d in v.items() if d})
    clusters = sorted(c for c, v in parts.items() if any(v.values()))
    s = np.zeros((len(worlds), len(clusters)))
    n = np.zeros((len(worlds), len(clusters)))
    wi = {w: i for i, w in enumerate(worlds)}
    for j, c in enumerate(clusters):
        for w, d in parts[c].items():
            if d:
                s[wi[w], j] = sum(d.values())
                n[wi[w], j] = len(d)
    return worlds, clusters, s, n


def one_way(s: np.ndarray, n: np.ndarray, draws: int = DRAWS) -> Est:
    """The world bootstrap of the pooled mean, clusters fixed (the two-way with one cluster)."""
    per_world = np.stack([s.sum(axis=1), n.sum(axis=1)], axis=1) if s.size else np.zeros((0, 2))
    per_world = per_world[per_world[:, 1] > 0]
    return ex.world_estimate(
        per_world, lambda t: t[..., 0] / t[..., 1], int(per_world[:, 1].sum()), draws
    )


# =========================================================================== threshold tests
def _test(key: str, value: float | None, meets: bool | None, threshold: str) -> dict[str, Any]:
    return {"key": key, "value": value, "meets_threshold": meets, "threshold": threshold}


def tested(tests: Sequence[Mapping[str, Any]] = (), **extra: Any) -> dict[str, Any]:
    """The comparisons of one entry of `threshold_tests` (none when it is not computed)."""
    return {"tests": list(tests), **extra}


def _get(node: Any, *keys: str) -> Any:
    """node[k1][k2]... or None where a level is missing."""
    for k in keys:
        if not isinstance(node, Mapping) or node.get(k) is None:
            return None
        node = node[k]
    return node


def _pp(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:+.1f} pp"


def _pc(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:.1f}%"


def _r(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.2f}"


def _iv(e: Mapping[str, Any] | None, f: Callable[[float | None], str] = _pp) -> str:
    if not _present(e):
        return "n/a"
    assert e is not None
    return f"{f(e['est'])} [{f(e['lo'])}, {f(e['hi'])}]"


def above_zero(key: str, e: Mapping[str, Any] | None) -> dict[str, Any]:
    """A test that the estimate is positive with its interval above 0."""
    if not _present(e):
        return _test(key, None, None, "est > 0 and 95% lower bound > 0")
    assert e is not None
    return _test(key, e["est"], bool(e["est"] > 0 and e["lo"] > 0), "est > 0 and lo > 0")


def cell_worlds(tab: ex.Table, systems: Iterable[str]) -> set[str]:
    """The worlds in which every one of `systems` answered."""
    sets = [
        {r["world_id"] for b in tab.branches for r in tab.rows(s, b) if r.get("cv") is not None}
        for s in systems
    ]
    return set.intersection(*sets) if sets else set()


def cover_worlds(entry: dict[str, Any], present: Iterable[str], key: str) -> dict[str, Any]:
    """`entry` with `worlds_missing`, the worlds of `BLOCK_WORLDS[key]` not among `present`, when
    its comparisons have values and a world is missing."""
    have = set(present)
    missing = [w for w in BLOCK_WORLDS[key] if w not in have]
    values = entry["tests"] and all(t["value"] is not None for t in entry["tests"])
    return {**entry, "worlds_missing": missing} if missing and values else entry


# ===================================================================================== block U
def ratio_quantity(x: Vec, b0: Vec, b1: Vec, draws: int = DRAWS) -> dict[str, Any]:
    """R = (Update(x) - Update(B0)) / (Update(B1) - Update(B0)) over the item-times all three
    cells answered, with the levels, numerator and denominator; R is None (not evaluated) when
    the denominator is below 5 pp."""
    keys = _common(x, b0, b1)
    xv, zv, ov = (_restrict(v, keys) for v in (x, b0, b1))
    out: dict[str, Any] = {
        "n_items": _n_items(keys),
        "n_worlds": len(keys),
        "update": {
            "x": ex.vec_mean(xv, draws),
            "B0": ex.vec_mean(zv, draws),
            "B1": ex.vec_mean(ov, draws),
        },
        "numerator": ex.paired_estimate(xv, zv, draws),
        "denominator": ex.paired_estimate(ov, zv, draws),
        "R": None,
        "evaluated": False,
    }
    den = out["denominator"]["est"]
    if not keys:
        out["reason"] = "no item-time answered in all three cells"
    elif den < MIN_DENOMINATOR - EPS:
        out["reason"] = f"denominator {_pp(den)} < 5 pp: not evaluated"
    else:
        out["R"] = ratio_estimate(item_diffs(xv, zv), item_diffs(ov, zv), draws)
        out["evaluated"] = True
    return out


def u_cell(arm: str, kind: str) -> str:
    return f"{ARMS[arm][0]}_{kind}"


def _arms_of(tn: ex.Table, kind: str) -> list[str]:
    return [a for a in ARMS if u_cell(a, kind) in tn.systems]


def u_family(tn: ex.Table, tc: ex.Table, draws: int = DRAWS) -> dict[str, Any]:
    """R_refresh and R_upsert per kind, branch, arm and class; the pooled ratios of U3; U4; the
    worlds the refresh cells (U1), the upsert cells (U3) and D1B0_upsert (U4, U5) answered in."""
    out: dict[str, Any] = {"R": {}, "pooled_definition_D1_Dm_src_def": {}, "U4": None}
    out["worlds"] = {
        "refresh": sorted(cell_worlds(tn, [u_cell(a, "refresh") for a in ("D1", "Dm")])),
        "upsert": sorted(cell_worlds(tn, [u_cell(a, "upsert") for a in ("D1", "Dm")])),
        "upsert_D1": sorted(cell_worlds(tn, [u_cell("D1", "upsert")])),
    }
    for kind in U_KINDS:
        node: dict[str, Any] = {}
        for b in BRANCHES:
            for arm in _arms_of(tn, kind):
                for cls in CLASSES:
                    cond = changed(cls)
                    x = tn.vec(u_cell(arm, kind), b, "cv", cond)
                    if not any(x.values()):
                        continue
                    lo, hi = ARMS[arm]
                    q = ratio_quantity(
                        x, tc.vec(lo, b, "cv", cond), tc.vec(hi, b, "cv", cond), draws
                    )
                    node.setdefault(b, {}).setdefault(arm, {})[cls] = q
        out["R"][kind] = node
        arms = [a for a in ("D1", "Dm") if a in _arms_of(tn, kind)]
        if arms:
            out["pooled_definition_D1_Dm_src_def"][kind] = pooled_ratio(tn, tc, kind, arms, draws)
    if u_cell("D1", "upsert") in tn.systems:
        cond = changed("all")
        x = tn.vec(u_cell("D1", "upsert"), "orphan", "cv", cond)
        z = tc.vec(ARMS["D1"][0], "orphan", "cv", cond)
        keys = _common(x, z)
        if keys:
            out["U4"] = {
                "difference": ex.paired_estimate(_restrict(x, keys), _restrict(z, keys), draws),
                "update_upsert": ex.vec_mean(_restrict(x, keys), draws),
                "update_B0": ex.vec_mean(_restrict(z, keys), draws),
            }
    return out


def pooled_ratio(
    tn: ex.Table, tc: ex.Table, kind: str, arms: Sequence[str], draws: int = DRAWS
) -> dict[str, Any]:
    """U3's ratio: the definition item-times of the (arm, S_src / S_def) combinations together."""
    cond = changed("definition")
    pooled: dict[str, Vec] = {
        "x": defaultdict(dict),
        "B0": defaultdict(dict),
        "B1": defaultdict(dict),
    }
    combinations = []
    for arm in arms:
        lo, hi = ARMS[arm]
        for b in SHIFTS:
            vs = {
                "x": tn.vec(u_cell(arm, kind), b, "cv", cond),
                "B0": tc.vec(lo, b, "cv", cond),
                "B1": tc.vec(hi, b, "cv", cond),
            }
            keys = _common(*vs.values())
            if keys:
                combinations.append(f"{arm}/{b}")
            for name, v in vs.items():
                for w, d in _restrict(v, keys).items():
                    pooled[name][w].update({f"{arm}|{b}|{k}": x for k, x in d.items()})
    q = ratio_quantity(pooled["x"], pooled["B0"], pooled["B1"], draws)
    q["combinations"] = combinations
    return q


def _u3(pooled: Mapping[str, Any] | None, need: Sequence[str]) -> dict[str, Any]:
    """U3's pooled ratio against its two bounds, when it pools every combination in `need`."""
    if not pooled:
        return tested()
    lacking = [c for c in need if c not in pooled.get("combinations", ())]
    r = pooled.get("R")
    if lacking or not pooled.get("evaluated") or r is None:
        return tested()
    t = _test("R_upsert_pooled", r["est"], None, ">= 0.8 executability; <= 0.5 key_schedule")
    return tested([t])


def _cover(entry: dict[str, Any], node: Any, cells: str, key: str) -> dict[str, Any]:
    """`cover_worlds` on the worlds `u_family` recorded for `cells` (none recorded: as is)."""
    worlds = _get(node, "worlds", cells)
    return entry if worlds is None else cover_worlds(entry, worlds, key)


def u_tests(main: Mapping[str, Any] | None, m2: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if main is None:
        return {k: tested() for k in ("U1", "U3", "U4", "U5")}
    # U1
    tests = []
    for arm in ("D1", "Dm"):
        for b in SHIFTS:
            q = _get(main, "R", "refresh", b, arm, "definition")
            key = f"R_refresh_{arm}_{b}_definition"
            if q is None:
                tests.append(_test(key, None, None, ">= 0.8 (cell not in the runs)"))
            elif not q["evaluated"]:
                tests.append({**_test(key, None, None, ">= 0.8"), "not_evaluated": q["reason"]})
            else:
                tests.append(_test(key, q["R"]["est"], q["R"]["est"] >= U1_MIN - EPS, ">= 0.8"))
    out["U1"] = _cover(tested(tests), main, "refresh", "U")
    # U3
    u3 = _u3(_get(main, "pooled_definition_D1_Dm_src_def", "upsert"), U3_COMBINATIONS)
    out["U3"] = _cover(u3, main, "upsert", "U")
    # U4
    u4 = main.get("U4")
    if u4 is None:
        out["U4"] = tested()
    else:
        t = above_zero("Update(D1B0_upsert) - Update(D1B0), S_del, all items", u4["difference"])
        out["U4"] = _cover(tested([t]), main, "upsert_D1", "U")
    # U5
    fams: dict[str, Any] = {}
    for name, res in sorted(m2.items()):
        u3 = _u3(_get(res, "pooled_definition_D1_Dm_src_def", "upsert"), U5_COMBINATIONS)
        fams[name] = _cover(u3, res, "upsert_D1", "U5")
    out["U5"] = tested(families=fams) if m2 else tested()
    return out


def _family_label(family: str) -> str:
    writer, reader, ks, months = json.loads(family)
    head = "main" if writer == MAIN_MODEL and reader == MAIN_MODEL else writer
    return head if (ks, months) == (None, 12) else f"{head}|{ks}|{months}"


def block_u(runs: Sequence[Path], compare: Sequence[Path], draws: int = DRAWS) -> dict[str, Any]:
    new, skipped = read_runs(runs)
    need = set(ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = read_runs(expand(compare), need.__contains__)
    newm, cmpm = Merged(new, by_family), Merged(cmp_runs, by_family)
    pair = pairing(newm, cmpm, [*skipped, *skipped_c], _scope_shared)
    fams: dict[str, Any] = {}
    for fam in newm.groups():
        fams[_family_label(fam)] = u_family(newm.table(fam), cmpm.table(fam), draws)
    main = fams.get("main")
    m2 = {k: v for k, v in fams.items() if k != "main" and "|" not in k}
    tests = u_tests(main, m2)
    reported = {
        "U2_R_refresh_D0_definition": {
            b: _get(main, "R", "refresh", b, "D0", "definition") for b in SHIFTS
        }
    }
    return _result("U", runs, compare, pair, {"families": fams}, tests, reported, NOTES["U"])


# ===================================================================================== block H
def _calls(run_dirs: Iterable[Path], model: str) -> dict[str, Row]:
    """Distinct write requests of `model` in the runs' calls.jsonl (key -> the logged call)."""
    out: dict[str, Row] = {}
    for d in run_dirs:
        for c in _rows(d / "calls.jsonl"):
            if c.get("model") == model and c.get("purpose") == "write":
                out.setdefault(str(c["key"]), c)
    return out


def _cache_check(path: Path, keys: Iterable[str]) -> dict[str, Any]:
    """Per distinct request, from the response cache: cut off by the output limit, or a reply
    that does not parse into a writer output (as `holdmem.writer.derive(reasoning=True)`)."""
    found = unreadable = at_limit = 0
    missing: list[str] = []
    stops: dict[str, int] = defaultdict(int)
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for key in sorted(keys):
            row = db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
            if row is None:
                missing.append(key)
                continue
            resp = json.loads(row[0])
            found += 1
            cut = resp.get("finish_reason") in CUT_OFF
            data = parse_json(str(resp.get("text") or ""))
            at_limit += cut
            unreadable += cut or not (isinstance(data, dict) and "statement" in data)
            stops[str((resp.get("api") or {}).get("stop_reason", resp.get("finish_reason")))] += 1
    finally:
        db.close()
    return {
        "requests": found,
        "missing_in_cache": len(missing),
        "unreadable": unreadable,
        "unreadable_share": unreadable / found if found else None,
        "at_limit": at_limit,
        "stop_reasons": dict(sorted(stops.items())),
    }


def _def_stats(run_dirs: Iterable[Path]) -> dict[str, int]:
    """`def_stats` of every cell, branch and world in the run summaries, summed."""
    tot = {"writes": 0, "unreadable": 0, "at_limit": 0}
    for d in run_dirs:
        for s in _rows(d / "world_summaries.jsonl"):
            for node in (s.get("branches") or {}).values():
                for st in (node.get("defwriter") or {}).values():
                    for k in tot:
                        tot[k] += int(st.get(k, 0))
    return tot


def h_validity(
    run_dirs: Sequence[Path], model: str, response_cache: Path | None = None
) -> dict[str, Any]:
    """The validity of a Claude definition-item writer (H-a, H-c): output-limit and unreadable
    shares of its write requests."""
    calls = _calls(run_dirs, model)
    n = len(calls)
    at_limit = sum(c.get("finish_reason") in CUT_OFF for c in calls.values())
    finish: dict[str, int] = defaultdict(int)
    for c in calls.values():
        finish[str(c.get("finish_reason"))] += 1
    tokens = sorted(int(c.get("completion_tokens") or 0) for c in calls.values())
    stats = _def_stats(run_dirs)
    out: dict[str, Any] = {
        "calls": {
            "distinct_write_requests": n,
            "at_limit": at_limit,
            "at_limit_share": at_limit / n if n else None,
            "finish_reasons": dict(sorted(finish.items())),
            "completion_tokens_mean": sum(tokens) / n if n else None,
            "completion_tokens_max": tokens[-1] if tokens else None,
        },
        "cell_writes": {
            **stats,
            "unreadable_share": stats["unreadable"] / stats["writes"] if stats["writes"] else None,
            "at_limit_share": stats["at_limit"] / stats["writes"] if stats["writes"] else None,
        },
        "cache": _cache_check(response_cache, calls) if response_cache and n else None,
    }
    if not n:
        out["within_limits"] = None
        out["reason"] = f"no write request of {model} in calls.jsonl"
        return out
    cache = out["cache"]
    if cache and cache["requests"] and not cache["missing_in_cache"]:
        unreadable, basis = cache["unreadable_share"], "distinct requests (response cache)"
    else:
        unreadable = out["cell_writes"]["unreadable_share"]
        basis = "cell writes (run summaries' def_stats)"
    share = out["calls"]["at_limit_share"]
    tests = [
        _test("at_limit_share", share, share <= MAX_AT_LIMIT + EPS, "<= 10%"),
        _test(
            "unreadable_share",
            unreadable,
            None if unreadable is None else unreadable <= MAX_UNREADABLE + EPS,
            "<= 2%",
        ),
    ]
    out["tests"] = tests
    out["unreadable_basis"] = basis
    meets = [t["meets_threshold"] for t in tests]
    out["within_limits"] = None if None in meets else all(meets)
    return out


def r0(tab: ex.Table, branch: str = "src", cls: str = "definition") -> dict[str, Any]:
    """r of Eq. (1) in D0B1_source: ContractValid of the changed item-times the key invalidated,
    and m, the share of changed item-times it did not invalidate."""
    system = ARMS["D0"][1]
    det = tab.vec(system, branch, "cv", _detected_changed(cls))
    flags = tab.vec(system, branch, "detected", changed(cls))
    return {
        "r0": ex.vec_mean(det),
        "m0": ex.vec_mean({w: {k: 1.0 - v for k, v in d.items()} for w, d in flags.items()}),
        "vec": det,
    }


def h_writer(
    tab: ex.Table,
    tc: ex.Table,
    run_dirs: Sequence[Path],
    model: str,
    response_cache: Path | None,
    draws: int = DRAWS,
) -> dict[str, Any]:
    """H-a or H-c: validity, B x D (H1), r_0 (H2) and the reported contrasts."""
    validity = h_validity(run_dirs, model, response_cache)
    worlds = sorted(cell_worlds(tab, [*ARMS["D0"], *ARMS["D1"]]))
    q: dict[str, Any] = {"validity": validity, "worlds": worlds, "bxd": {}}
    for b in BRANCHES:
        for cls in CLASSES:
            for arm in ("D1", "Dm"):
                cells = {ARMS["D0"][0], ARMS["D0"][1], *ARMS[arm]}
                if not cells <= tab.systems:
                    continue
                e = bxd(getter(tab, b, changed(cls)), arm)
                q["bxd"].setdefault(b, {}).setdefault(cls, {})[arm] = e
    mine, ref = r0(tab), r0(tc)
    keys = _common(mine["vec"], ref["vec"])
    q["r0"] = {
        "writer": mine["r0"],
        "writer_m0": mine["m0"],
        "qwen_same_worlds": ex.vec_mean({w: d for w, d in ref["vec"].items() if w in mine["vec"]}),
        "paired_difference": ex.paired_estimate(
            _restrict(mine["vec"], keys), _restrict(ref["vec"], keys), draws
        ),
        "qwen_main_study": QWEN_R0,
    }
    return q


def h_tests(model: str, q: Mapping[str, Any]) -> dict[str, Any]:
    """H1 and H2 of one definition-item writer, computed only when its writes are within the
    validity limits; for Claude Sonnet 5.5 also its r_0 against 70% (`sonnet_r0_below_70pct`)."""
    within = q["validity"].get("within_limits")
    out: dict[str, Any] = {}
    e = ((q["bxd"].get("def") or {}).get("definition") or {}).get("D1")
    if within and _present(e):
        assert e is not None
        t = _test(
            "BxD_def_definition_D1", e["est"], None, "<= 10 pp or CI incl. 0; >= 20 pp, CI > 0"
        )
        out["H1"] = tested([t])
    else:
        out["H1"] = tested()
    r = q["r0"]["writer"]
    if within and _present(r):
        gap = r["est"] - QWEN_R0
        t = _test("r0_minus_qwen_main_study", gap, gap >= H2_MARGIN - EPS, ">= 20 pp")
        out["H2"] = tested([t])
    else:
        out["H2"] = tested()
    if "worlds" in q:
        out = {k: cover_worlds(v, q["worlds"], "H") for k, v in out.items()}
    if model.startswith("claude-sonnet") and _present(r):
        missing = [w for w in BLOCK_WORLDS["H"] if w not in q.get("worlds", ())]
        out["sonnet_r0_below_70pct"] = {
            "r0": r["est"],
            "below_70pct": bool(r["est"] < SONNET_R0_BOUND),
            "on_all_worlds": "worlds" not in q or not missing,
            "threshold": "r_0 of Claude Sonnet 5.5 on changed definition items after S_src < 70%",
        }
    return out


def think_validity(runs: Sequence[Run]) -> dict[str, Any]:
    """H-b: the thinking writes after the shift (item notes of D0thinkB1_source): cut off (the
    completion tokens reach the run's limit) and unreadable (`note.think.ok` false)."""
    writes = cut = unreadable = 0
    limits = sorted({r.think_max_tokens or 0 for r in runs})
    for run in runs:
        limit = run.think_max_tokens
        for it in run.items:
            if it["system"] != THINK_CELL:
                continue
            for h in it["history"]:
                note = h.get("think")
                if note is None:
                    continue
                writes += 1
                unreadable += not note["ok"]
                cut += bool(limit) and int(note["completion_tokens"]) >= int(limit or 0)
    cut_share = cut / writes if writes else None
    unread_share = unreadable / writes if writes else None
    tests = [
        _test(
            "cut_off_share",
            cut_share,
            None if cut_share is None else cut_share <= MAX_THINK_CUT + EPS,
            "<= 10%",
        )
    ]
    return {
        "limits": limits,
        "writes": writes,
        "cut_off": cut,
        "cut_off_share": cut_share,
        "unreadable": unreadable,
        "unreadable_share": unread_share,
        "unreadable_at_most_20pct": None
        if unread_share is None
        else unread_share <= MAX_THINK_UNREADABLE + EPS,
        "tests": tests,
        "within_limits": tests[0]["meets_threshold"],
    }


def h_think(tab: ex.Table, tc: ex.Table, runs: Sequence[Run], draws: int = DRAWS) -> dict[str, Any]:
    validity = think_validity(runs)
    gaps: dict[str, Any] = {}
    for metric in ("repair", "repair_mech"):
        for b in BRANCHES:
            x = mvec(tc, ARMS["D1"][1], b, metric, "definition")
            y = mvec(tab, THINK_CELL, b, metric, "definition")
            gaps.setdefault(metric, {})[b] = ex.paired_estimate(x, y, draws)
    tests = [
        _test(
            f"{m}_{b}",
            gaps[m][b]["est"] if _present(gaps[m][b]) else None,
            gaps[m][b]["est"] >= H3_MIN - EPS if _present(gaps[m][b]) else None,
            ">= 30 pp",
        )
        for m in ("repair", "repair_mech")
        for b in SHIFTS
    ]
    h3 = tested(tests) if validity["within_limits"] else tested()
    worlds = sorted(cell_worlds(tab, [THINK_CELL]))
    h3 = cover_worlds(h3, worlds, "H")
    return {"validity": validity, "worlds": worlds, "D1_minus_think": gaps, "H3": h3}


def block_h(
    runs: Sequence[Path],
    compare: Sequence[Path],
    response_cache: Path | None = None,
    draws: int = DRAWS,
) -> dict[str, Any]:
    new, skipped = read_runs(runs, items=lambda s: True)
    need = set(ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = read_runs(
        expand(compare), need.__contains__, {ARMS["D0"][1], ARMS["D1"][1]}.__contains__
    )
    newm, cmpm = Merged(new, by_variant), Merged(cmp_runs, by_family)
    pair = pairing(newm, cmpm, [*skipped, *skipped_c], _scope_h)
    quantities: dict[str, Any] = {"writers": {}, "think": None}
    thresholds: dict[str, Any] = {}
    ignored = []
    committed: dict[str, ex.Table] = {}
    for var in newm.groups():
        vruns = newm.runs_of(var)
        head = vruns[0]
        if head.family not in committed:
            committed[head.family] = cmpm.table(head.family)
        tc = committed[head.family]
        if head.defwriter:
            q = h_writer(
                newm.table(var), tc, [r.path for r in vruns], head.defwriter, response_cache, draws
            )
            quantities["writers"][head.defwriter] = q
            tag = "" if head.defwriter.startswith("claude-sonnet") else f"_{head.defwriter}"
            for k, v in h_tests(head.defwriter, q).items():
                thresholds[f"{k}{tag}"] = v
        elif THINK_CELL in head.cells:
            th = h_think(newm.table(var), tc, vruns, draws)
            quantities["think"] = th
            thresholds["H3"] = th["H3"]
        else:
            ignored += [r.name for r in vruns]
    for k in ("H1", "H2", "H3"):
        thresholds.setdefault(k, tested())  # no run of this arm given
    quantities["ignored_runs"] = ignored
    sonnet_r0 = thresholds.pop("sonnet_r0_below_70pct", None)
    reported = {"sonnet_r0_below_70pct": sonnet_r0}
    return _result("H", runs, compare, pair, quantities, thresholds, reported, NOTES["H"])


# ===================================================================================== block G
def _arm_vecs(get: Callable[[str], Vec], arm: str) -> Vec:
    lo, hi = ARMS[arm]
    return double_diff(get(hi), get(lo), get(ARMS["D0"][1]), get(ARMS["D0"][0]))


def _by_fact(v: Vec) -> dict[str, Vec]:
    out: dict[str, Vec] = defaultdict(lambda: defaultdict(dict))
    for w, d in v.items():
        for k, x in d.items():
            out[k.split("@")[0]][w][k] = x
    return out


def g_heldout(tables: Mapping[str, ex.Table], draws: int = DRAWS) -> dict[str, Any]:
    """G1, G2: per branch and arm, B x D on the definition items of the held-out draws, two-way
    over worlds x definitions (a definition is a draw's definition: `heldout:<seed>|k.<id>`)."""
    out: dict[str, Any] = {}
    cond = changed("definition")
    for b in BRANCHES:
        for arm in ("D1", "Dm"):
            parts: dict[str, Vec] = {}
            for ks, tab in sorted(tables.items()):
                if not {*ARMS["D0"], *ARMS[arm]} <= tab.systems:
                    continue
                dd = _arm_vecs(getter(tab, b, cond), arm)
                for fact, v in _by_fact(dd).items():
                    parts[f"{ks}|{fact}"] = v
            if not parts:
                continue
            worlds, clusters, s, n = cluster_arrays(parts)
            out.setdefault(b, {})[arm] = {
                "two_way": two_way(s, n, draws),
                "one_way_worlds": one_way(s, n, draws),
                "definitions_with_items": clusters,
                "per_definition": {c: ex.vec_mean(parts[c], draws) for c in clusters},
            }
    return out


def _writer_share(values: Mapping[str, float], writer: Mapping[str, str]) -> dict[str, Any]:
    """Share of the variance of the per-set values that lies between writers: eta squared
    (between-writer over total sum of squares) and the one-way random-effects share."""
    sets = [s for s in sorted(values) if not math.isnan(values[s])]
    groups: dict[str, list[float]] = defaultdict(list)
    for s in sets:
        groups[writer[s]].append(values[s])
    n, k = len(sets), len(groups)
    if k < 2 or n <= k:
        return {"sets": n, "writers": k, "eta_squared": None, "variance_component_share": None}
    grand = sum(values[s] for s in sets) / n
    ss_t = sum((values[s] - grand) ** 2 for s in sets)
    ss_b = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values())
    ms_b, ms_w = ss_b / (k - 1), (ss_t - ss_b) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups.values()) / n) / (k - 1)
    var_b = max(0.0, (ms_b - ms_w) / n0)
    return {
        "sets": n,
        "writers": k,
        "eta_squared": ss_b / ss_t if ss_t > 0 else None,
        "variance_component_share": var_b / (var_b + ms_w) if var_b + ms_w > 0 else None,
    }


def g_programs(
    sets: Mapping[str, tuple[str, ex.Table]],
    tc: ex.Table,
    draws: int = DRAWS,
    classes: Sequence[str] = ("definition", "all"),
) -> dict[str, Any]:
    """G3 (and V1, V2) and their reported quantities: B x D at D_m with each program set's D1s
    cells against the committed D=0 cells, two-way over worlds x program sets. Definition-only
    runs answer no other item after the fork: pass `classes=("definition",)`."""
    out: dict[str, Any] = {"branches": {}}
    writer = {s: w for s, (w, _) in sets.items()}
    for b in BRANCHES:
        for cls in classes:
            cond = changed(cls)
            parts: dict[str, Vec] = {}
            for sid, (_, tab) in sorted(sets.items()):
                if not set(ARMS["Dm"]) <= tab.systems:
                    continue

                def get(s: str, tab: ex.Table = tab, cond: Cond = cond, b: str = b) -> Vec:
                    return (tab if s in ARMS["Dm"] else tc).vec(s, b, "cv", cond)

                parts[sid] = _arm_vecs(get, "Dm")
            if not parts:
                continue
            worlds, clusters, s, n = cluster_arrays(parts)
            per_writer: dict[str, Vec] = defaultdict(lambda: defaultdict(dict))
            for sid, v in parts.items():
                for w, d in v.items():
                    per_writer[writer[sid]][w].update({f"{sid}|{k}": x for k, x in d.items()})
            out["branches"].setdefault(b, {})[cls] = {
                "two_way": two_way(s, n, draws),
                "one_way_worlds": one_way(s, n, draws),
                "sets_with_items": clusters,
                "per_set": {c: ex.vec_mean(parts[c], draws) for c in clusters},
                "per_writer": {w: ex.vec_mean(v, draws) for w, v in sorted(per_writer.items())},
            }
    node = (out["branches"].get("def") or {}).get("definition")
    if node:
        values = {s: float(e["est"]) for s, e in node["per_set"].items()}
        out["writer_variance_share"] = _writer_share(values, writer)
    out["sets"] = {s: {"writer": w} for s, w in sorted(writer.items())}
    return out


def _set_id(run: Run) -> str:
    """A program set: writer, temperature and sample index (V's sets pair with G-b's on it)."""
    assert run.program is not None
    model, temp, sample = run.program
    return f"{model}#t{temp:g}#s{sample}"


def _classes(runs: Iterable[Run]) -> tuple[str, ...]:
    """The classes a set of program-set runs answers after the fork."""
    return ("definition",) if any(r.definition_only for r in runs) else ("definition", "all")


def block_g(runs: Sequence[Path], compare: Sequence[Path], draws: int = DRAWS) -> dict[str, Any]:
    new, skipped = read_runs(runs)
    need = set(ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = read_runs(expand(compare), need.__contains__)
    newm, cmpm = Merged(new, by_variant), Merged(cmp_runs, by_family)
    pair = pairing(newm, cmpm, [*skipped, *skipped_c], _scope_g)
    heldout: dict[str, ex.Table] = {}
    programs: dict[str, tuple[str, ex.Table]] = {}
    families: set[str] = set()
    ignored = []
    for var in newm.groups():
        head = newm.runs_of(var)[0]
        if head.kappa_set and head.kappa_set.startswith("heldout:"):
            if head.kappa_set in heldout:
                raise ValueError(f"two variants of {head.kappa_set}: pass one set of runs")
            heldout[head.kappa_set] = newm.table(var)
        elif head.program is not None and not head.user_words:
            programs[_set_id(head)] = (head.program[0], newm.table(var))
            families.add(head.family)
        else:  # block V's sets (user's words) are computed by `block_v`
            ignored += [r.name for r in newm.runs_of(var)]
    if len(families) > 1:
        raise ValueError("program sets of different writer families: pass one family")
    ga = g_heldout(heldout, draws) if heldout else {}
    prog_runs = [r for r in new if r.program is not None and not r.user_words]
    gb = (
        g_programs(programs, cmpm.table(families.pop()), draws, _classes(prog_runs))
        if programs
        else {}
    )
    thresholds: dict[str, Any] = {}
    cells_a = [*ARMS["D0"], *ARMS["D1"], *ARMS["Dm"]]
    worlds_a = [cell_worlds(t, cells_a) for t in heldout.values()]
    worlds_a_all = set.intersection(*worlds_a) if worlds_a else set()
    for key, b in (("G1", "def"), ("G2", "src")):
        if len(heldout) < G_DRAWS:  # fewer than the 3 held-out draws (18 definitions)
            thresholds[key] = tested()
            continue
        node = ga.get(b) or {}
        tests = [
            above_zero(f"BxD_{b}_definition_{arm}_two_way", _get(node, arm, "two_way"))
            for arm in ("D1", "Dm")
        ]
        thresholds[key] = cover_worlds(tested(tests), worlds_a_all, "G-a")
    g3 = ((gb.get("branches") or {}).get("def") or {}).get("definition")
    if not programs or g3 is None or len(g3["sets_with_items"]) < G3_MIN_SETS:
        thresholds["G3"] = (
            tested()
        )  # fewer than 2 program sets with definition item-times after S_def
    else:
        g3_pred = tested([above_zero("BxD_def_definition_Dm_two_way", g3["two_way"])])
        worlds_b = [cell_worlds(t, ARMS["Dm"]) for _, t in programs.values()]
        thresholds["G3"] = cover_worlds(g3_pred, set.intersection(*worlds_b), "G-b")
        thresholds["G3"]["sets"] = len(g3["sets_with_items"])
        thresholds["G3"]["sets_total"] = G_SETS
    sdel = (gb.get("branches") or {}).get("orphan") or {}
    reported = {
        "writer_variance_share": gb.get("writer_variance_share"),
        "S_del_interaction_per_writer": {
            cls: (sdel.get(cls) or {}).get("per_writer") for cls in ("definition", "all")
        },
    }
    quantities = {
        "heldout": ga,
        "programs": gb,
        "heldout_draws": sorted(heldout),
        "program_sets": sorted(programs),
        "ignored_runs": ignored,
    }
    return _result("G", runs, compare, pair, quantities, thresholds, reported, NOTES["G"])


# ===================================================================================== block Q
def block_q(runs: Sequence[Path], compare: Sequence[Path], draws: int = DRAWS) -> dict[str, Any]:
    new, skipped = read_runs(runs)
    need = set(ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = read_runs(expand(compare), need.__contains__)
    unnamed = [r for r in new if r.unnamed]
    newm, cmpm = Merged(unnamed, by_family), Merged(cmp_runs, by_family)
    pair = pairing(newm, cmpm, [*skipped, *skipped_c], _scope_shared)
    if len(unnamed) < len(new):
        pair["detail"]["not_unnamed_runs_ignored"] = [r.name for r in new if not r.unnamed]
    q: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    fams = [f for f in newm.groups() if _family_label(f) == "main"]
    if not fams:
        thresholds = {k: tested() for k in ("Q1", "Q2")}  # no unnamed-probe run of the main writer
        return _result("Q", runs, compare, pair, q, thresholds, {}, NOTES["Q"])
    tq, tc = newm.table(fams[0]), cmpm.table(fams[0])
    cond = changed("definition")
    lo, hi = ARMS["D1"]
    bu = item_diffs(tq.vec(hi, "def", "cv", cond), tq.vec(lo, "def", "cv", cond))
    bn = item_diffs(tc.vec(hi, "def", "cv", cond), tc.vec(lo, "def", "cv", cond))
    keys = _common(bu, bn)
    q["Q1"] = {
        "B_effect_D1_unnamed": ex.vec_mean(_restrict(bu, keys), draws),
        "B_effect_D1_named": ex.vec_mean(_restrict(bn, keys), draws),
        "difference": ex.paired_estimate(_restrict(bu, keys), _restrict(bn, keys), draws),
    }
    q["Q2"] = {}
    for arm in ("D1", "Dm"):
        if {*ARMS["D0"], *ARMS[arm]} <= tq.systems:
            q["Q2"][arm] = {
                "unnamed": bxd(getter(tq, "def", cond), arm),
                "named": bxd(getter(tc, "def", cond), arm),
            }
    d = q["Q1"]["difference"]
    thresholds["Q1"] = tested(
        [
            _test(
                "B_effect_D1_unnamed_minus_named",
                d["est"] if _present(d) else None,
                d["est"] >= Q1_MARGIN - EPS if _present(d) else None,
                ">= 5 pp",
            )
        ]
    )
    thresholds["Q2"] = tested(
        [above_zero("BxD_def_definition_D1_unnamed", (q["Q2"].get("D1") or {}).get("unnamed"))]
    )
    q["worlds"] = sorted(cell_worlds(tq, [*ARMS["D0"], *ARMS["D1"]]))
    thresholds = {k: cover_worlds(v, q["worlds"], "Q") for k, v in thresholds.items()}
    reported = {"Q2_Dm": q["Q2"].get("Dm")}
    return _result("Q", runs, compare, pair, q, thresholds, reported, NOTES["Q"])


# ===================================================================================== block N
def n_anchor(natural: Merged, stage_runs: Sequence[Path]) -> dict[str, int]:
    """The natural branch's k = 0 answers of the four cells against the committed S_def
    branch's, per (cell, world, item)."""
    cells = {f"holdmem_{c}" for c in natmod.PLAN_CELLS}
    mine: dict[tuple[str, str, str], Row] = {}
    for (_, system, w, b), run in natural.owner.items():
        if system in cells and b == "natural":
            for a in run.by_cell[(system, w, b)].values():
                if _k(a) == 0:
                    mine[(system, w, a["fact_id"])] = a
    worlds = {w for _, w, _ in mine}
    theirs: dict[tuple[str, str, str], Row] = {}
    for d in stage_runs:
        for a in _rows(d / "answers.jsonl", cells.__contains__):
            if a["branch"] == "def" and _k(a) == 0 and a["world_id"] in worlds:
                theirs[(a["system"], a["world_id"], a["fact_id"])] = a
    keys = mine.keys() | theirs.keys()
    keys = {k for k in keys if k[1] in worlds}
    missing = sum(k not in mine or k not in theirs for k in keys)
    differ = sum(
        k in mine and k in theirs and any(mine[k].get(f) != theirs[k].get(f) for f in ANSWER_FIELDS)
        for k in keys
    )
    return {"compared": len(keys) - missing, "differ": differ, "missing": missing}


def n_tests(cf: Mapping[str, Any]) -> dict[str, Any]:
    tests = []
    for b in SHIFTS:
        node = cf[b]["shift"]["definition"]
        t = above_zero(f"paired_difference_{b}_shift_definition", node["difference"])
        if node["item_times_with_natural"] < node["item_times_in_branch"]:
            t = {
                **t,
                "meets_threshold": None,
                "coverage": (node["item_times_with_natural"], node["item_times_in_branch"]),
            }
        tests.append(t)
    return {"N1": tested(tests)}


def block_n(runs: Sequence[Path], stage: Path) -> dict[str, Any]:
    cells = {f"holdmem_{c}" for c in natmod.PLAN_CELLS}
    new, skipped = read_runs(runs, cells.__contains__)
    newm = Merged(new, by_family)
    pair: dict[str, Any] = {
        "detail": {
            "incomplete_runs_skipped": skipped,
            "superseded": newm.superseded,
            "calls": {run.name: call_counts(run) for run in new},
        }
    }
    counts: dict[str, int] = {
        "superseded_answers_differ": sum(s["differ"] + s["missing"] for s in newm.superseded),
        **_hash(new),
        "discriminative_runs": sum(r.params.get("probes_after_fork") != "all" for r in new),
        "runs_missing_cells": sum(not cells <= set(r.cells) for r in new),
    }
    stage_runs = r2._source_runs(stage) or []
    anchor = n_anchor(newm, stage_runs)
    counts["anchor_answers_differ"] = anchor["differ"]
    counts["anchor_answers_missing"] = anchor["missing"]
    counts["runs_without_anchor"] = int(anchor["compared"] == 0)
    pair.update(
        ok=not any(counts.values()), counts=counts, anchor_answers_compared=anchor["compared"]
    )
    lab = natmod.label_item_times(stage)
    tab = ex.Table(r2._jsonl(stage / "analysis" / "e1_item_table.jsonl"))
    nat: dict[str, Vec] = defaultdict(lambda: defaultdict(dict))
    for (_, system, w, b), run in newm.owner.items():
        if b != "natural":
            continue
        for a in run.by_cell[(system, w, b)].values():
            if _k(a) > 0:
                nat[system][w][f"{a['fact_id']}@{_k(a)}"] = float(a["contract_valid"])
    cf = {
        b: {
            part: {
                cls: natmod.paired_vs_natural(tab, nat, b, lab.cause, part, cls)
                for cls in (*natmod.CLASSES, "all")
            }
            for part in natmod.PARTS
        }
        for b in BRANCHES
    }
    thresholds = n_tests(cf)
    reported = {
        "other_items": {b: cf[b]["shift"]["other"]["difference"] for b in BRANCHES},
        "all_items": {b: cf[b]["shift"]["all"]["difference"] for b in BRANCHES},
        "S_del_definition": cf["orphan"]["shift"]["definition"]["difference"],
        "natural_also_changed": natural_moved(tab, newm, lab.cause),
    }
    return _result(
        "N", runs, [stage], pair, {"counterfactual": cf}, thresholds, reported, NOTES["N"]
    )


def natural_moved(tab: ex.Table, natural: Merged, cause: Mapping[Any, str]) -> dict[str, Any]:
    """Descriptive, no threshold: among the item-times the paired difference uses (changed by
    the shift, asked in all four shifted-branch cells), the share whose gold on the natural branch
    also differs from the natural branch's own t0 gold (the item moved without the shift too)."""
    gold: dict[tuple[str, str, int], str] = {}
    for run in natural.runs:
        for p in run.probes:
            if p["branch"] == "natural":
                gold[p["world_id"], p["fact_id"], _k(p)] = r2._gold(p)
    out: dict[str, Any] = {}
    for b in BRANCHES:
        for cls in (*natmod.CLASSES, "all"):
            quad = [
                tab.vec(ex.cell_system("D1", d, bb, "source"), b, "cv", changed(cls))
                for d in (0, 1)
                for bb in (0, 1)
            ]
            keys = _common(*quad)
            moved: Vec = defaultdict(dict)
            for w, ks in keys.items():
                for key in ks:
                    fact, k = key.split("@")
                    if cause.get((w, b, fact, int(k))) != "shift":
                        continue
                    now, then = gold.get((w, fact, int(k))), gold.get((w, fact, 0))
                    if now is not None and then is not None:
                        moved[w][key] = float(now != then)
            out.setdefault(b, {})[cls] = ex.vec_mean(moved)
    return out


# ===================================================================================== block Y
def y_validity(runs: Sequence[Run], model: str) -> dict[str, Any]:
    """Y validity: at most 2% of the reader's answers unreadable (reader status "error") or cut
    off at the output limit (finish_reason "length" in calls.jsonl), counted separately over every
    answer the reader gave (a cut-off answer that is also unreadable counts twice)."""
    answers = [a for r in runs for a in r.answers if a.get("status") != "blocked"]
    unreadable = sum(a.get("status") == "error" for a in answers)
    arm_answers = [a for a in answers if a["system"] in ARM_CELLS]
    calls = [
        c
        for r in runs
        for c in _rows(r.path / "calls.jsonl")
        if c.get("model") == model and c.get("purpose") == "serve"
    ]
    cut = sum(c.get("finish_reason") in CUT_OFF for c in calls)
    finish: dict[str, int] = defaultdict(int)
    for c in calls:
        finish[str(c.get("finish_reason"))] += 1
    n = len(answers)
    share = (unreadable + cut) / n if n else None
    test = _test(
        "unreadable_or_cut_off_share",
        share,
        None if share is None else share <= MAX_UNREADABLE + EPS,
        "<= 2%",
    )
    return {
        "readers": sorted({r.reader for r in runs}),
        "answers": n,
        "answers_arm_cells": len(arm_answers),
        "unreadable": unreadable,
        "unreadable_arm_cells": sum(a.get("status") == "error" for a in arm_answers),
        "reader_calls": len(calls),
        "reader_calls_cached": sum(bool(c.get("cached")) for c in calls),
        "cut_off": cut,
        "finish_reasons": dict(sorted(finish.items())),
        "share": share,
        "tests": [test],
        "within_limits": test["meets_threshold"],
    }


def store_requests_sent(runs: Sequence[Run]) -> int:
    """Write and program requests a Y run sent (its stores must be the committed ones: 0)."""
    return sum(
        1
        for r in runs
        for c in _rows(r.path / "calls.jsonl")
        if c.get("purpose") in ("write", "program") and not c.get("cached")
    )


def y_contrasts(ty: ex.Table, tq: ex.Table, draws: int = DRAWS) -> dict[str, Any]:
    """Per branch, class and D=1 arm: B x D with the swapped-in reader on its own item-times,
    B x D with the committed reader, and their paired per-item difference over the item-times
    both answered (world bootstrap)."""
    out: dict[str, Any] = {}
    for b in BRANCHES:
        for cls in CLASSES:
            for arm in ("D1", "Dm"):
                cells = {*ARMS["D0"], *ARMS[arm]}
                if not (cells <= ty.systems and cells <= tq.systems):
                    continue
                x = _arm_vecs(getter(ty, b, changed(cls)), arm)
                y = _arm_vecs(getter(tq, b, changed(cls)), arm)
                keys = _common(x, y)
                out.setdefault(b, {}).setdefault(cls, {})[arm] = {
                    "reader": ex.vec_mean(x, draws),
                    "committed_reader": ex.vec_mean(y, draws),
                    "difference": ex.paired_estimate(_restrict(x, keys), _restrict(y, keys), draws),
                    "n_items": _n_items(x),
                    "n_items_committed": _n_items(y),
                    "n_items_paired": _n_items(keys),
                }
    return out


def y_store_reader(data: Mapping[str, list[Row]], draws: int = DRAWS) -> dict[str, Any]:
    """`ex_round3_store_reader` on the swapped-in reader's answers (the same stores): the store
    and reader terms per arm, and the right stores the reader lost, by gold status."""
    rows = e1.item_table(dict(data))
    store = srmod.Store(dict(data))
    res = {arm: srmod.analyse_arm(store, rows, arm, draws) for arm, _ in srmod.ARMS}
    lost: dict[str, Any] = {}
    for arm, node in res.items():
        for cls in ("definition", "other", "all"):
            tot = {"right_stores": 0, "lost": 0, "right_abstain": 0, "lost_abstain": 0}
            for b in BRANCHES:
                c = node[b][cls]["counts"]["D1B1"]
                part = {
                    "right_stores": c["ds"],
                    "lost": c["ds"] - c["dsc"],
                    "right_abstain": c["ds_abs"],
                    "lost_abstain": c["ds_abs"] - c["dsc_abs"],
                }
                lost.setdefault(arm, {}).setdefault(cls, {})[b] = part
                tot = {k: tot[k] + v for k, v in part.items()}
            lost[arm][cls]["shifts"] = tot
    return {"results": res, "lost_at_D1B1": lost}


def y_tests(q: Mapping[str, Any]) -> dict[str, Any]:
    con = q["contrasts"]
    out: dict[str, Any] = {}
    for key, b in (("Y1", "def"), ("Y2", "orphan")):
        e = _get(con, b, "other", "D1", "reader")
        t = above_zero(f"BxD_{b}_other_D1_reader", e)
        out[key] = tested([t])
    tests = []
    for b in SHIFTS:
        d = _get(con, b, "definition", "D1", "difference")
        value = float(d["est"]) if _present(d) else None
        meets = None if value is None else abs(value) <= Y3_MARGIN + EPS
        key = f"BxD_{b}_definition_D1_reader_minus_committed"
        tests.append(_test(key, value, meets, "|est| <= 5 pp"))
    out["Y3"] = tested(tests)
    return {k: cover_worlds(v, q["worlds"], "Y") for k, v in out.items()}


Y_CELLS = frozenset((*ARMS["D0"], *ARMS["D1"]))  # the cells Y1-Y3 use


def note_contexts(
    tests: dict[str, Any], pair: Mapping[str, Any], context_cells: Iterable[str]
) -> dict[str, Any]:
    """A note on every comparison when the only pairing count is a context hash that differs in
    cells no Y comparison uses (Y1-Y3 use the authored cells D0B0 to D1B1_source); those answers
    are left out of every quantity."""
    cells = set(context_cells)
    counts = {k: v for k, v in pair.get("counts", {}).items() if v}
    if not counts or set(counts) - {"context_hash_differs"} or cells & Y_CELLS or not cells:
        return tests
    note = (
        f"{counts['context_hash_differs']} answers whose context differs from the committed one, "
        f"all in {', '.join(sorted(cells))}, which no Y comparison uses; left out of every "
        "quantity"
    )
    return {k: {**v, "pairing_note": note} for k, v in tests.items()}


def block_y(runs: Sequence[Path], compare: Sequence[Path], draws: int = DRAWS) -> dict[str, Any]:
    new, skipped = read_runs(runs, items=lambda s: True)
    need = set(ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = read_runs(expand(compare), need.__contains__)
    newm, cmpm = Merged(new, by_variant), Merged(cmp_runs, by_family)
    pair = pairing(newm, cmpm, [*skipped, *skipped_c], _scope_context, by_store_family)
    sent = store_requests_sent(new)
    pair["counts"]["store_requests_sent"] = sent
    pair["ok"] = not any(pair["counts"].values())
    groups = [g for g in newm.groups() if newm.runs_of(g)[0].reader != newm.runs_of(g)[0].writer]
    if len(groups) != 1:
        raise ValueError(f"Y: {len(groups)} swapped-in reader variants given: pass one")
    var = groups[0]
    yruns = newm.runs_of(var)
    model = yruns[0].reader
    # an answer whose context differs from the committed one did not read the committed store:
    # it is left out of every quantity (its item-time drops out of the cell's contrasts)
    listed = _get(pair, "detail", "context_differs", "answers") or []
    differing = {(s, w, b, pid) for s, w, b, pid in listed}
    data = newm.data(var)
    data["answers"] = [
        a
        for a in data["answers"]
        if (a["system"], a["world_id"], a["branch"], a["probe_id"]) not in differing
    ]
    ty = ex.Table(e1.item_table(data))
    tq = cmpm.table(yruns[0].store_family)
    q: dict[str, Any] = {
        "reader": model,
        "validity": y_validity(yruns, model),
        "worlds": sorted(cell_worlds(ty, Y_CELLS)),
        "contrasts": y_contrasts(ty, tq, draws),
        "answers_left_out_context_differs": len(differing),
    }
    thresholds = note_contexts(y_tests(q), pair, {s for s, *_ in differing})
    sr = y_store_reader(data, draws)
    con = q["contrasts"]
    reported = {
        "Dm": {
            b: {cls: _get(con, b, cls, "Dm") for cls in ("definition", "other")} for b in BRANCHES
        },
        "other_items_S_src_D1": _get(con, "src", "other", "D1"),
        "store_reader_interaction": {
            arm: {
                b: {cls: node[b][cls]["interaction"] for cls in ("definition", "other", "all")}
                for b in BRANCHES
            }
            for arm, node in sr["results"].items()
        },
        "right_no_value_stores_misread": {
            arm: {cls: v["shifts"] for cls, v in node.items()}
            for arm, node in sr["lost_at_D1B1"].items()
        },
    }
    q["store_reader"] = sr
    return _result("Y", runs, compare, pair, q, thresholds, reported, NOTES["Y"])


# ===================================================================================== block V
def _program_t0(run: Run) -> Vec:
    """Per definition slot of the run's D1sB1_source cell: does the program's value at t0 (the
    `derive` event, before any shift) equal the gold (`e1_synth.program_stats`'s agree_t0)?"""
    gold0 = {(p["world_id"], p["branch"], p["fact_id"]): p for p in run.probes if _k(p) == 0}
    out: Vec = defaultdict(dict)
    for it in run.items:
        if it["system"] != ARMS["Dm"][1] or not str(it["slot_id"]).startswith("k."):
            continue
        for event in it["history"]:
            note = event.get("program")
            if note is None or event["event"] != "derive":
                continue
            p = gold0[it["world_id"], it["branch"], it["slot_id"]]
            ok = bool(note["ok"]) and e1._repair_matches(p, note["value"])
            out[it["world_id"]][f"{it['branch']}|{it['slot_id']}"] = float(ok)
            break
    return out


def _pool(parts: Mapping[str, Vec]) -> Vec:
    """Per-set vectors pooled into one (keys prefixed with the set)."""
    out: Vec = defaultdict(dict)
    for sid, v in parts.items():
        for w, d in v.items():
            out[w].update({f"{sid}|{k}": x for k, x in d.items()})
    return out


def v_versus_gb(
    v_sets: Mapping[str, tuple[str, ex.Table, list[Run]]],
    gb_sets: Mapping[str, tuple[str, ex.Table, list[Run]]],
    draws: int = DRAWS,
) -> dict[str, Any]:
    """Per writer, V minus G-b over the sets of the same writer and sample index: Update on the
    changed definition item-times at D1sB1_source (per branch and the three pooled), and the
    programs' agreement with gold at t0 (paired per item, world bootstrap)."""
    out: dict[str, Any] = {}
    paired = sorted(s for s in v_sets if s in gb_sets)
    writers = sorted({v_sets[s][0] for s in paired})
    cell = ARMS["Dm"][1]
    for wr in writers:
        sids = [s for s in paired if v_sets[s][0] == wr]
        node: dict[str, Any] = {"sets": sids, "update_D1sB1_source": {}}
        for group in (*BRANCHES, "pooled"):
            bs = BRANCHES if group == "pooled" else (group,)
            vv = {
                f"{s}|{b}": v_sets[s][1].vec(cell, b, "cv", changed("definition"))
                for s in sids
                for b in bs
            }
            gg = {
                f"{s}|{b}": gb_sets[s][1].vec(cell, b, "cv", changed("definition"))
                for s in sids
                for b in bs
            }
            x, y = _pool(vv), _pool(gg)
            keys = _common(x, y)
            node["update_D1sB1_source"][group] = {
                "V": ex.vec_mean(_restrict(x, keys), draws),
                "G_b": ex.vec_mean(_restrict(y, keys), draws),
                "difference": ex.paired_estimate(_restrict(x, keys), _restrict(y, keys), draws),
            }
        tv = _pool({s: _merge_vecs(_program_t0(r) for r in v_sets[s][2]) for s in sids})
        tg = _pool({s: _merge_vecs(_program_t0(r) for r in gb_sets[s][2]) for s in sids})
        keys = _common(tv, tg)
        node["agreement_t0"] = {
            "V": ex.vec_mean(_restrict(tv, keys), draws),
            "G_b": ex.vec_mean(_restrict(tg, keys), draws),
            "difference": ex.paired_estimate(_restrict(tv, keys), _restrict(tg, keys), draws),
        }
        out[wr] = node
    return out


def _merge_vecs(vecs: Iterable[Vec]) -> Vec:
    out: Vec = defaultdict(dict)
    for v in vecs:
        for w, d in v.items():
            out[w].update(d)
    return out


def block_v(runs: Sequence[Path], compare: Sequence[Path], draws: int = DRAWS) -> dict[str, Any]:
    """V1, V2 on the V program sets (user's words); the per-writer comparison with the G-b sets
    of the same writer and sample, when their runs are given among `runs` too."""
    new, skipped = read_runs(runs, items=lambda s: s == ARMS["Dm"][1])
    need = set(ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = read_runs(expand(compare), need.__contains__)
    newm, cmpm = Merged(new, by_variant), Merged(cmp_runs, by_family)
    pair = pairing(newm, cmpm, [*skipped, *skipped_c], _scope_g)
    v_sets: dict[str, tuple[str, ex.Table, list[Run]]] = {}
    gb_sets: dict[str, tuple[str, ex.Table, list[Run]]] = {}
    families: set[str] = set()
    ignored = []
    for var in newm.groups():
        vruns = newm.runs_of(var)
        head = vruns[0]
        if head.program is None or head.kappa_set:
            ignored += [r.name for r in vruns]
            continue
        target = v_sets if head.user_words else gb_sets
        target[_set_id(head)] = (head.program[0], newm.table(var), vruns)
        families.add(head.family)
    if len(families) > 1:
        raise ValueError("program sets of different writer families: pass one family")
    q: dict[str, Any] = {
        "V_sets": sorted(v_sets),
        "G_b_sets": sorted(gb_sets),
        "ignored_runs": ignored,
        "programs": {},
        "V_minus_G_b": {},
    }
    thresholds: dict[str, Any] = {}
    if v_sets:
        tc = cmpm.table(families.pop())
        vruns_all = [r for _, _, rs in v_sets.values() for r in rs]
        q["programs"] = g_programs(
            {s: (w, t) for s, (w, t, _) in v_sets.items()}, tc, draws, _classes(vruns_all)
        )
        q["V_minus_G_b"] = v_versus_gb(v_sets, gb_sets, draws)
    worlds = (
        set.intersection(*(cell_worlds(t, ARMS["Dm"]) for _, t, _ in v_sets.values()))
        if v_sets
        else set()
    )
    for key, b in (("V1", "def"), ("V2", "src")):
        node = _get(q["programs"], "branches", b, "definition")
        if not v_sets:
            thresholds[key] = tested()
            continue
        t = above_zero(f"BxD_{b}_definition_Dm_two_way", node and node["two_way"])
        thresholds[key] = cover_worlds(tested([t]), worlds, "V")
        thresholds[key]["sets"] = len(v_sets)
    sdel = _get(q["programs"], "branches", "orphan", "definition") or {}
    reported = {
        "V_minus_G_b_per_writer": q["V_minus_G_b"],
        "S_del_interaction": {"two_way": sdel.get("two_way"), "per_writer": sdel.get("per_writer")},
    }
    return _result("V", runs, compare, pair, q, thresholds, reported, NOTES["V"])


# ===================================================================================== block X
def x1_tests(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """X1: per cell with a diagnosis and both effects, whether the arm that targets the dominant
    failure wins (point estimates); `n_cells`, `n_favour_target` and `n_required` (at least 4,
    or every cell when fewer than 4 are compared; at least 3 cells)."""
    base = xa.x1(cells)
    n, fav = base["n_cells"], base["n_favour_target"]
    tests = [
        _test(
            name,
            c["target_effect"] - c["other_effect"],
            c["favours_target"],
            f"{c['target']} > other arm",
        )
        for name, c in sorted(base["cells"].items())
    ]
    if n < X_MIN_CELLS:
        return tested(tests, n_cells=n)
    return tested(tests, n_cells=n, n_favour_target=fav, n_required=min(X_MIN_FAVOUR, n))


def x_tests(
    cells: Mapping[str, Mapping[str, Any]], worlds: Iterable[str], pair: Mapping[str, Any]
) -> dict[str, Any]:
    """X1 (`x1_tests`) on the worlds present; not computed in a void run."""
    if not pair.get("ok"):
        return {"X1": tested()}
    return {"X1": cover_worlds(x1_tests(cells), worlds, "X")}


def _newest_x_runs(runs: Sequence[Run]) -> tuple[list[Run], list[str]]:
    """The newest run per world; a run that is newest for only part of its worlds is an error."""
    worlds: dict[str, list[Run]] = defaultdict(list)
    covers: dict[str, set[str]] = {}
    for run in runs:
        ws = {str(r["world"]["world_id"]) for r in _rows(run.path / "world_results.jsonl")}
        covers[run.name] = ws
        for w in ws:
            worlds[w].append(run)
    keep: dict[str, Run] = {}
    for _, rs in sorted(worlds.items()):
        best = max(rs, key=lambda r: (r.created, r.name))
        keep[best.name] = best
    for run in keep.values():
        lost = [
            w
            for w in covers[run.name]
            if max(worlds[w], key=lambda r: (r.created, r.name)) is not run
        ]
        if lost:
            raise ValueError(
                f"{run.name} is superseded for {sorted(lost)} only: pass runs per world"
            )
    dropped = [r.name for r in runs if r.name not in keep]
    return list(keep.values()), dropped


def block_x(runs: Sequence[Path], draws: int = DRAWS) -> dict[str, Any]:
    new, skipped = read_runs(runs, lambda s: False)
    chosen, dropped = _newest_x_runs(new)
    with tempfile.TemporaryDirectory() as tmp:
        res = xa.analyse([r.path for r in chosen], Path(tmp), draws)
    anchors = res.get("anchor") or {}
    differ = sum(int(a["compared"]) - int(a["answer_identical"]) for a in anchors.values() if a)
    counts = {"anchor_answers_differ": differ}  # X stores have no context-hash assertion
    pair = {
        "ok": not any(counts.values()),
        "counts": counts,
        "anchor_answers_compared": sum(int(a["compared"]) for a in anchors.values() if a),
        "detail": {
            "incomplete_runs_skipped": skipped,
            "superseded_runs": dropped,
            "worlds_without_anchor": sorted(w for w, a in anchors.items() if not a),
            "prefix_sources": res.get("prefix_sources"),
            "anchor_per_world": anchors,
            # reported, not a pairing count: the anchor rule is on answers
            "anchor_contexts_differ": sum(
                int(a["compared"]) - int(a["context_identical"]) for a in anchors.values() if a
            ),
        },
    }
    thresholds = x_tests(res["cells"], res.get("worlds") or [], pair)
    reported = {
        "m_r_a_and_effects": {
            name: {
                "arms": c.get("arms"),
                "diagnosis": c.get("diagnosis"),
                "effects": c.get("effects"),
            }
            for name, c in res["cells"].items()
        },
        "support": res.get("support"),
        "mutations_after_fork": res.get("mutations_after_fork"),
    }
    return _result("X", runs, [], pair, {"cells": res["cells"]}, thresholds, reported, NOTES["X"])


# ====================================================================================== output
NOTES = {
    "U": ["U1", "U3", "U4", "U5"],
    "H": ["H1", "H2", "H", "H3"],
    "G": ["G1-G3"],
    "Q": ["Q1"],
    "N": ["N1"],
    "X": ["X1"],
    "Y": ["Y", "Y1-Y3"],
    "V": ["V1-V2", "V"],
}


def _notes(keys: Sequence[str]) -> dict[str, str]:
    """The paragraphs of the module docstring that state these comparisons."""
    doc = __doc__ or ""
    part = doc.split("Comparisons and their thresholds", 1)[-1]
    out: dict[str, str] = {}
    for k in keys:
        m = re.search(rf"^  {re.escape(k)} +(.*?)(?=^  \S|\Z)", part, re.S | re.M)
        if m:
            out[k] = " ".join(m.group(1).split())
    return out


def _result(
    block: str,
    runs: Sequence[Path],
    compare: Sequence[Path],
    pair: Mapping[str, Any],
    quantities: Mapping[str, Any],
    tests: Mapping[str, Any],
    reported: Mapping[str, Any],
    notes: Sequence[str],
) -> dict[str, Any]:
    return {
        "block": block,
        "status": STATUS,
        "units": UNITS,
        "inputs": {"runs": [str(p) for p in runs], "compare": [str(p) for p in compare]},
        "pairing": dict(pair),
        "threshold_tests": dict(tests),
        "reported": dict(reported),
        "quantities": dict(quantities),
        "notes": _notes(notes),
        "code": code_state(),
    }


# ===================================================================================== summary
def tests_lines(entries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """One line per entry of `threshold_tests`: each comparison's key, value, threshold and
    whether the value meets it."""
    lines = []
    for name, p in entries.items():
        shown = [
            f"{t['key']} {t['value'] if t['value'] is None else round(t['value'], 4)} "
            f"({t['threshold']}: {t['meets_threshold']})"
            for t in p["tests"]
        ]
        lines.append(f"  {name}: {'; '.join(shown) or 'not computed'}")
    return lines


def summary(res: Mapping[str, Any]) -> str:
    """A compact human-readable account of a block's result."""
    pair = res["pairing"]
    bad = {k: v for k, v in pair.get("counts", {}).items() if v}
    lines = [
        f"block {res['block']}: pairing {'ok' if pair.get('ok') else 'FAILED ' + json.dumps(bad)}"
        f" ({pair.get('anchor_answers_compared', 0)} anchor answers compared)"
    ]
    lines += tests_lines(res["threshold_tests"])
    lines += _detail_lines(res)
    return "\n".join(lines)


def _detail_lines(res: Mapping[str, Any]) -> list[str]:
    q, out = res["quantities"], []
    if res["block"] == "U":
        for fam, node in q.get("families", {}).items():
            for kind, by_b in node["R"].items():
                for b, by_arm in by_b.items():
                    for arm, by_cls in by_arm.items():
                        d = by_cls.get("definition")
                        if not d:
                            continue
                        u = d["update"]
                        r = d["R"]["est"] if d["R"] else None
                        out.append(
                            f"  {fam} {kind:7} {b:6} {arm} definition: Update B0/x/B1 "
                            f"{_pc(u['B0']['est'])}/{_pc(u['x']['est'])}/{_pc(u['B1']['est'])}"
                            f"  R {_r(r)}  (n {d['n_items']}, {d['n_worlds']} worlds)"
                        )
    elif res["block"] == "H":
        for model, node in q.get("writers", {}).items():
            v = node["validity"]
            out.append(
                f"  {model}: within limits {v.get('within_limits')}  "
                f"r0 {_iv(node['r0']['writer'], _pc)}"
            )
    elif res["block"] == "G":
        for b, by_arm in (q.get("heldout") or {}).items():
            for arm, node in by_arm.items():
                out.append(f"  heldout {b} {arm}: two-way {_iv(node['two_way'])}")
        for b, by_cls in ((q.get("programs") or {}).get("branches") or {}).items():
            node = by_cls.get("definition")
            if node:
                out.append(f"  programs {b} Dm definition: two-way {_iv(node['two_way'])}")
    elif res["block"] == "Q":
        if q.get("Q1"):
            out.append(f"  Q1 difference {_iv(q['Q1']['difference'])}")
    elif res["block"] == "N":
        for b, node in q.get("counterfactual", {}).items():
            out.append(f"  {b} shift definition: {_iv(node['shift']['definition']['difference'])}")
    elif res["block"] == "Y":
        v = q["validity"]
        out.append(
            f"  {q['reader']}: {v['answers']} answers, {v['unreadable']} unreadable, "
            f"{v['cut_off']} cut off (within limits {v['within_limits']})"
        )
        for b, by_cls in q.get("contrasts", {}).items():
            for cls, by_arm in by_cls.items():
                for arm, node in by_arm.items():
                    out.append(
                        f"  {b:6} {cls:10} {arm}: B x D reader {_iv(node['reader'])}, committed "
                        f"{_iv(node['committed_reader'])}, difference {_iv(node['difference'])}"
                    )
    elif res["block"] == "V":
        for b, by_cls in ((q.get("programs") or {}).get("branches") or {}).items():
            node = by_cls.get("definition")
            if node:
                out.append(f"  V programs {b} Dm definition: two-way {_iv(node['two_way'])}")
    return out


# ========================================================================================= CLI
BLOCKS = ("U", "H", "G", "Q", "N", "X", "Y", "V")


def run_block(
    block: str,
    runs: Sequence[Path],
    compare: Sequence[Path],
    response_cache: Path | None = None,
    draws: int = DRAWS,
) -> dict[str, Any]:
    if block == "U":
        return block_u(runs, compare, draws)
    if block == "H":
        return block_h(runs, compare, response_cache, draws)
    if block == "G":
        return block_g(runs, compare, draws)
    if block == "Q":
        return block_q(runs, compare, draws)
    if block == "N":
        if len(compare) != 1:
            raise SystemExit("N: --compare takes the main E1 stage directory (one path)")
        return block_n(runs, compare[0])
    if block == "X":
        return block_x(runs, draws)
    if block == "Y":
        return block_y(runs, compare, draws)
    if block == "V":
        return block_v(runs, compare, draws)
    raise SystemExit(f"unknown block {block!r}")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 2)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("block", choices=BLOCKS)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--compare", nargs="*", default=[], type=Path)
    ap.add_argument("--response-cache", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = run_block(a.block, a.runs, a.compare, a.response_cache, a.draws)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(summary(res))
    if not res["pairing"].get("ok"):
        noted = any("pairing_note" in p for p in res["threshold_tests"].values())
        msg = (
            "PAIRING: contexts differ only in cells no comparison uses (pairing.detail)"
            if noted
            else "PAIRING CHECK FAILED: the run is void (see pairing.counts)"
        )
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
