"""Paper numbers and tables of the registered round-3 blocks (prereg/v4_round3.md blocks U, N, H,
Q and G, and block X, which is void; prereg/v4b_round3_claude.md blocks Y and V). The verdicts are
those of `analysis.round3_registered`, whose JSON results are staged in runs/stage7_round3/analysis/
(named in results/sources.json as `round3`; the stage manifest lists the commands, the
source runs and the response-cache snapshots; `python -m memshift.analysis.round3_paper` reruns
them from the raw runs):

  u.json         block U on the main writer: the refresh and upsert cells (U1-U4), and U5 on the
                 upsert runs of the three M2 families (paired with the M2 stages)
  u_delta.json   exploratory (`analysis.ex_round3_upsert`): where the upsert can act, i.e. whether
                 the B=0 delta derivation yielded a new item for the slot of each changed item-time
  n.json         block N, the natural-branch counterfactual (N1 and its unthresholded reports)
  y.json         block Y, Claude Sonnet 5.5 as reader of the committed stores (Y1-Y3, validity, the
                 store/reader split with that reader)
  h_<writer>.json  block H per definition-item writer (`H_WRITERS`; H-a: Claude Sonnet 5.5,
                 h_sonnet.json; H-c: Claude Opus 5.5, h_opus.json, run because Sonnet's r_0 was
                 below 70%), judged from that writer's runs and its response-cache snapshot (H1,
                 H2, validity)
  h_b.json       H-b, the thinking Qwen3.5-9B writer at a 32,768-token limit (H3; its registered
                 validity rule decides whether H3 is judged)
  q.json         block Q, probes that do not name the definition version (Q1, Q2 and Q2's D_m)
  g_a.json       block G on the held-out-definition runs (G-a: G1, G2, two-way intervals over
                 worlds x the 18 definitions)
  g_b.json       block G on the 18 program sets of six writers (G-b: G3, two-way intervals over
                 worlds x program sets; the writer share of the per-set variance)
  v.json         block V, programs from the definition in the user's own words (V1, V2, two-way
                 intervals over worlds x the nine program sets), with the G-b sets of the same
                 writers and samples, which only its reported per-writer comparison (V minus G-b)
                 reads
  x.json         block X, diagnosis and interventions on Mem0, one run per world. The run is void:
                 its anchor answers differ from E5's committed ones, so X1 is not judged (and not
                 computed, `round3_registered.x_predictions`); per decisions.md 2026-10-17 the
                 cells (m, r, a per class and branch, the two arms' effects) are reported as
                 exploratory. The anchor counts stay in this file: no macro or table states them

The committed reader's store/reader split is the exploratory one of `analysis.ex_round3` (stage
`ex_round3`); its store parts equal block Y's, because both read the same stores.

Macros (letters only; a stage that is not staged renders as the MISSING marker, a quantity over no
item-time or not evaluated as a dash):
  RThree<pred>Verdict                   holds, fails, judged or not judged, for UOne, UThree, UFour,
                                        UFive, NOne, YOne, YTwo, YThree, GThree (from g_b.json) and
                                        HThree; UTwo (no threshold): reported. UThreeReading: the
                                        reading U3 triggers
  RThreeUFive<family>...                U5 per M2 family (Gemma, Llama, GptOss): the pooled upsert
                                        ratio R, its item-times and its reading; UFiveSame: the
                                        families with the main writer's U3 reading
  RThreeU<arm><branch><class>...        arm Dzero, Done, Dm: R of the refresh cell (ratio), its
                                        Update levels (B=0, refresh, B=1), the gain over B=0 and the
                                        binding effect it is a share of, item-times; ...Upsert...:
                                        the same for the upsert cell
  RThreeUThree..., RThreeUFour...       U3's pooled ratio with its levels; U4's difference
  RThreeUDelta<arm><branch><class><part>...  part Yielded or None: item-times, share, Update of
                                        B=0, upsert and B=1, share of binding's effect, answers the
                                        upsert moved above or below B=0
  RThreeN<branch><class>...             B x D on the item-times the shift changed, on the same
                                        item-times of the natural branch, the paired difference, the
                                        share of them that also changed on the natural branch
  RThreeY<branch><class><arm>...        B x D with the Claude reader and with the committed reader
                                        (arm Done, Dm), their paired difference, item-times
  RThreeYSR<arm><branch><class>...      the store/reader split with the Claude reader; per cell its
                                        r, r_store, a, a_store; counts of right stores it lost
  RThreeY...                            validity counts, the context anchor
  RThreeH<writer>...                    writer Sonnet or Opus: H1 (verdict, reading), H2
                                        (verdict); B x D per branch, class and arm (Done, Dm); r_0,
                                        Qwen3.5-9B's r_0 on the same worlds and the paired
                                        difference; validity counts; worlds; anchor answers compared
  RThreeQ...                            Q1's B effect at D=1 with unnamed and named probes and their
                                        paired difference; B x D per arm (Done, Dm) with each;
                                        item-times, worlds, anchor answers compared (QOne, QTwo:
                                        the verdicts)
  RThreeGA<branch><arm>...              G-a: B x D on definition items with the two-way interval
                                        (BxD) and the one-way world interval (BxDOneWay),
                                        item-times, definitions with item-times and those whose
                                        own B x D is above 0; worlds and draws (GOne, GTwo: the
                                        verdicts)
  RThreeGB<branch>...                   G-b: B x D at D_m on definition items with the two-way
                                        interval (BxD) and the one-way world interval, item-times
                                        and program sets; RThreeGB<writer><branch>BxD per writer
                                        (six); GBDefWriterEtaSq: eta squared of the writer over the
                                        18 per-set B x D after S_def (GBDefWriterVarShare: the
                                        one-way random-effects share); worlds, writers, anchors
  RThreeHThink...                       H-b: the token limit, thinking writes, those cut off at the
                                        limit and unreadable, with shares (H3's validity); worlds
  RThreeV<branch>...                    V: B x D at D_m on definition items with the two-way
                                        interval (BxD) and the one-way world interval, item-times
                                        and program sets; RThreeV<writer>...: that writer's B x D
                                        per branch, and V minus G-b (Diff) with V's level in Update
                                        on changed definition items at D1sB1_source (per branch and
                                        pooled) and in agreement with gold at t0 (VOne, VTwo: the
                                        verdicts)
  RThreeX<branch><class>...             X (void, exploratory): changed item-times; native Mem0's
                                        m, r, a; the invalidation arm's m, r (Det); each arm's
                                        effect on Update (DetEffect, RepEffect); worlds; store
                                        mutations after the fork per arm (Native, Det); no anchor
                                        count (XOne: the verdict, not judged)

tab_round3   appendix: the judged predictions with their tests, values and verdicts or readings,
             the levels behind U1-U3 and the natural-branch test
tab_reader   appendix: B x D on Update by reader, class and shift, with the store and reader parts
tab_heldout  appendix: the 18 held-out definitions (`gen.kappa.heldout_draw`) with their own B x D
tab_userwords  appendix: block V per program writer
tab_mem0x    appendix: block X per branch and class (void, exploratory)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memshift.analysis.ex_e1_paper import UNDEFINED, _cell, fmt
from memshift.analysis.ex_round2 import BRANCH_LABELS
from memshift.analysis.paper_data import BRANCH_TAGS, FAMILIES, MISSING, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, pct, pp
from memshift.analysis.paper_tables import _num, _row, interval

STAGE_KEY = "round3"
STAGE_DIR = "runs/stage7_round3"
COMPARE = ("runs/stage2_e1", "runs/stage4_c_e1")  # the committed stages the blocks pair with
U_UPSERT_RUNS = (
    "runs/e1_u_upsert_s1000-1002_20261016T092158Z",
    "runs/e1_u_upsert_s1003-1005_20261016T110050Z",
    "runs/e1_u_upsert_s1006-1008_20261016T110521Z",
    "runs/e1_u_upsert_s1009-1011_20261016T110651Z",
    "runs/e1_u_upsert_s1012-1014_20261016T112722Z",
    "runs/e1_u_upsert_s1015-1017_20261016T115554Z",
    "runs/e1_u_upsert_s1018-1019_20261016T121025Z",
)
# U5: the upsert runs of the three M2 families on worlds 1000-1009 (stamps 20261016T......Z), paired
# with the M2 stages
U5_STAMPS = {
    "gemma": ("202715", "202845", "203015", "203145", "203315"),
    "gptoss": ("205928", "210058", "210228", "210358", "210528"),
    "llama": ("200247", "200417", "200547", "200717", "200847"),
}
PAIR_WORLDS = ("1000-1001", "1002-1003", "1004-1005", "1006-1007", "1008-1009")
U5_RUNS = tuple(
    f"runs/e1_u5_{family}_s{worlds}_20261016T{stamp}Z"
    for family, stamps in U5_STAMPS.items()
    for worlds, stamp in zip(PAIR_WORLDS, stamps, strict=True)
)
M2_COMPARE = (
    "runs/stage4_m2_google_gemma-4-12b-it",
    "runs/stage4_m2_meta-llama_llama-3_1-8b-instruct",
    "runs/stage4_m2_openai_gpt-oss-20b",
)
# H-b (H3): the thinking Qwen3.5-9B writer at a 32,768-token limit (stamps 20261016T......Z)
H_B_RUNS = tuple(
    f"runs/e1_hb_think32k_s{worlds}_20261016T{stamp}Z"
    for worlds, stamp in zip(
        PAIR_WORLDS, ("211635", "211805", "211935", "212105", "212235"), strict=True
    )
)


@dataclass(frozen=True)
class HWriter:
    """One definition-item writer of block H: its staged file, macro tag, model (the key of its
    quantities in the block result), prediction-name suffix, runs and response-cache snapshot."""

    key: str
    tag: str
    model: str
    suffix: str
    runs: tuple[str, ...]
    response_cache: str


H_WRITERS = (
    HWriter(
        "h_sonnet",
        "Sonnet",
        "claude-sonnet-5-5",
        "",  # round3_registered names Sonnet's predictions H1, H2 and another writer's H1_<model>
        tuple(
            f"runs/e1_ha_sonnet_s{s}_20261016T082038Z"
            for s in ("1000-1001", "1002-1003", "1004-1005", "1006-1007", "1008-1009")
        ),
        "runs/_snapshots/anthropic_claude-sonnet-5-5_20261016T1335Z.sqlite",
    ),
    HWriter(
        "h_opus",
        "Opus",
        "claude-opus-5-5",
        "_claude-opus-5-5",
        (
            "runs/e1_hc_opus_s1000-1001_20261016T115624Z",
            "runs/e1_hc_opus_s1002-1003_20261016T121055Z",
            "runs/e1_hc_opus_s1004-1005_20261016T123426Z",
            "runs/e1_hc_opus_s1006-1007_20261016T123857Z",
            "runs/e1_hc_opus_s1008-1009_20261016T124727Z",
        ),
        "runs/_snapshots/anthropic_claude-opus-5-5_20261016T1546Z.sqlite",
    ),
)
G_A_SEEDS = (4001, 4002, 4003)  # the sampler seeds of the three held-out draws (v4 §G-a)
# G-a: per draw, its four runs on worlds 1000-1009 (stamps 20261017T......Z; re-runs from the
# response cache, decisions.md 2026-10-17)
G_A_STAMPS = {
    4001: ("022248", "022419", "022549", "022719"),
    4002: ("022849", "023019", "023149", "023319"),
    4003: ("023449", "023619", "023749", "023919"),
}
G_A_RUNS = tuple(
    f"runs/e1_ga_heldout{seed}_s{worlds}_20261017T{stamp}Z"
    for seed, stamps in G_A_STAMPS.items()
    for worlds, stamp in zip(
        ("1000-1002", "1003-1005", "1006-1007", "1008-1009"), stamps, strict=True
    )
)
SET_WORLDS = ("1000-1003", "1004-1007", "1008-1011", "1012-1015", "1016-1019")  # one set's runs


def _set_runs(kind: str, stamps: Mapping[tuple[str, int], Sequence[str]]) -> tuple[str, ...]:
    """The five runs of each program set (writer, sample), stamps 2026MMDDTHHMMSSZ."""
    return tuple(
        f"runs/e1_{kind}_{writer}_p{sample}_s{worlds}_2026{stamp}Z"
        for (writer, sample), five in stamps.items()
        for worlds, stamp in zip(SET_WORLDS, five, strict=True)
    )


# G-b (G3): the 18 program sets of six writers, three samples each; the Qwen3.5-9B, Claude Sonnet
# 5.5 and Claude Opus 5.5 sets are re-runs from the response cache (decisions.md 2026-10-17)
G_B_STAMPS = {  # (writer, sample) -> the stamps (MMDDTHHMMSS) of its five runs
    ("gemma", 0): ("1016T214712", "1016T214842", "1016T215012", "1016T215242", "1016T215812"),
    ("gemma", 1): ("1016T215943", "1016T220113", "1016T220743", "1016T220913", "1016T221043"),
    ("gemma", 2): ("1016T221513", "1016T221644", "1016T221914", "1016T222044", "1016T222614"),
    ("gptoss", 0): ("1016T222744", "1016T222914", "1016T223144", "1016T223514", "1016T224645"),
    ("gptoss", 1): ("1016T231718", "1016T231847", "1016T232017", "1016T232148", "1016T233018"),
    ("gptoss", 2): ("1016T234949", "1016T235120", "1016T235250", "1016T235820", "1017T000650"),
    ("llama", 0): ("1016T211640", "1016T211810", "1016T211940", "1016T212110", "1016T212240"),
    ("llama", 1): ("1016T212411", "1016T212541", "1016T212711", "1016T212841", "1016T213411"),
    ("llama", 2): ("1016T213541", "1016T213711", "1016T213841", "1016T214011", "1016T214141"),
    ("opus", 0): ("1016T215133", "1016T215403", "1016T220034", "1016T220204", "1016T220334"),
    ("opus", 1): ("1016T220704", "1016T220834", "1016T221004", "1016T221134", "1016T221404"),
    ("opus", 2): ("1016T221934", "1016T222105", "1016T222535", "1016T222705", "1016T222835"),
    ("qwen", 0): ("1016T223105", "1016T223235", "1016T223405", "1016T223635", "1016T225006"),
    ("qwen", 1): ("1016T231838", "1016T232008", "1016T232138", "1016T233009", "1016T234940"),
    ("qwen", 2): ("1016T235110", "1016T235240", "1016T235410", "1016T235941", "1017T000811"),
    ("sonnet", 0): ("1017T000941", "1017T001111", "1017T001242", "1017T001412", "1017T001542"),
    ("sonnet", 1): ("1017T001712", "1017T001842", "1017T002012", "1017T002142", "1017T002312"),
    ("sonnet", 2): ("1017T002442", "1017T002612", "1017T002742", "1017T002912", "1017T003042"),
}
G_B_RUNS = _set_runs("gb", G_B_STAMPS)
# Block V: the nine program sets from the user's words (re-runs from the response cache,
# decisions.md 2026-10-17), then the G-b sets of the same writers and samples that its per-writer
# comparison pairs with
V_STAMPS = {  # (writer, sample) -> the stamps (MMDDTHHMMSS) of its five runs
    ("qwen", 0): ("1017T005443", "1017T005614", "1017T005744", "1017T005914", "1017T010044"),
    ("qwen", 1): ("1017T010214", "1017T010344", "1017T010514", "1017T010644", "1017T014617"),
    ("qwen", 2): ("1017T014747", "1017T014917", "1017T015047", "1017T015217", "1017T015347"),
    ("sonnet", 0): ("1017T015717", "1017T015947", "1017T020318", "1017T020448", "1017T020618"),
    ("sonnet", 1): ("1017T020748", "1017T020918", "1017T021048", "1017T021218", "1017T021348"),
    ("sonnet", 2): ("1017T021518", "1017T021648", "1017T021818", "1017T021948", "1017T022118"),
    ("opus", 0): ("1017T003212", "1017T003343", "1017T003513", "1017T003643", "1017T003813"),
    ("opus", 1): ("1017T003943", "1017T004113", "1017T004243", "1017T004413", "1017T004543"),
    ("opus", 2): ("1017T004713", "1017T004843", "1017T005013", "1017T005143", "1017T005313"),
}
V_RUNS = (
    *_set_runs("v", V_STAMPS),
    *_set_runs("gb", {key: G_B_STAMPS[key] for key in V_STAMPS}),
)
# the program writers of block V: (the model name of its sets, macro tag)
V_WRITERS = (
    ("Qwen/Qwen3.5-9B", "Qwen"),
    ("claude-sonnet-5-5", "Sonnet"),
    ("claude-opus-5-5", "Opus"),
)
# the program writers of block G-b: V's three, then the three M2 families (model name, macro tag)
G_B_WRITERS = (
    *V_WRITERS,
    *((f.model, f.tag) for k in ("gemma", "llama", "gptoss") for f in FAMILIES if f.key == k),
)
X_STAMPS = (  # block X: one run per world 1000-1009 (20261016T......Z)
    "165445",
    "165615",
    "165845",
    "170416",
    "170546",
    "170716",
    "170846",
    "171016",
    "171146",
    "171416",
)
X_RUNS = tuple(
    f"runs/x_mem0_s{seed}_20261016T{stamp}Z"
    for seed, stamp in zip(range(1000, 1010), X_STAMPS, strict=True)
)
FILES = {
    "u": "u.json",
    "u_delta": "u_delta.json",
    "n": "n.json",
    "y": "y.json",
    **{h.key: f"{h.key}.json" for h in H_WRITERS},
    "h_b": "h_b.json",
    "q": "q.json",
    "g_a": "g_a.json",
    "g_b": "g_b.json",
    "v": "v.json",
    "x": "x.json",
}
RUNS = {
    "u": (
        *(
            f"runs/e1_u_refresh_s{s}_20261016T060737Z"
            for s in ("1000-1003", "1004-1007", "1008-1011", "1012-1015", "1016-1019")
        ),
        *U_UPSERT_RUNS,
        *U5_RUNS,
    ),
    "u_delta": U_UPSERT_RUNS,
    "n": tuple(
        f"runs/e1_n_natural_s{s}_20261016T094158Z"
        for s in ("1000-1004", "1005-1009", "1010-1014", "1015-1019")
    ),
    "y": (
        "runs/e1_y_reader_s1000_20261016T095546Z",
        *(
            f"runs/e1_y_reader_s{s}_20261016T105417Z"
            for s in (
                "1001-1002",
                "1003-1004",
                "1005-1006",
                "1007-1008",
                "1009-1010",
                "1011-1012",
                "1013-1014",
                "1015-1016",
                "1017-1018",
                "1019",
            )
        ),
    ),
    **{h.key: h.runs for h in H_WRITERS},
    "h_b": H_B_RUNS,
    "q": tuple(
        f"runs/e1_q_unnamed_s{s}"
        for s in (
            "1000-1002_20261016T154710Z",
            "1003-1005_20261016T160742Z",
            "1006-1008_20261016T162213Z",
            "1009-1011_20261016T162343Z",
            "1012-1014_20261016T163113Z",
            "1015-1017_20261016T163844Z",
            "1018-1019_20261016T164915Z",
        )
    ),
    "g_a": G_A_RUNS,
    "g_b": G_B_RUNS,
    "v": V_RUNS,
    "x": X_RUNS,
}
BLOCKS = {
    "u": "U",
    "n": "N",
    "y": "Y",
    **{h.key: "H" for h in H_WRITERS},
    "h_b": "H",
    "q": "Q",
    "g_a": "G",
    "g_b": "G",
    "v": "V",
    "x": "X",
}
CACHES = {h.key: h.response_cache for h in H_WRITERS}


def _compare(key: str) -> tuple[str, ...]:
    """The committed runs a block pairs with: U also the M2 stages (U5), N the main stage only, X
    none (its anchor is the E5 answers each run recorded)."""
    return {"u": (*COMPARE, *M2_COMPARE), "n": COMPARE[:1], "x": ()}.get(key, COMPARE)


def _command(key: str) -> str:
    module = "ex_round3_upsert" if key == "u_delta" else f"round3_registered {BLOCKS[key]}"
    compare = ["--compare", *_compare(key)] if _compare(key) else []
    cache = [f"--response-cache {CACHES[key]}"] if key in CACHES else []
    return " ".join(
        [
            f"python -m memshift.analysis.{module}",
            "--runs",
            *RUNS[key],
            *compare,
            *cache,
            f"--out {STAGE_DIR}/analysis/{FILES[key]}",
        ]
    )


PRODUCERS = {f"analysis/{name}": _command(key) for key, name in FILES.items()}
NOTE = (
    "registered (prereg/v4_round3.md blocks U, N, Q, H and G; prereg/v4b_round3_claude.md blocks "
    "Y and V): verdicts of analysis.round3_registered on the raw runs listed in each command, no "
    "model call. u.json: block U, the main writer's refresh and upsert cells and U5 on the three "
    "M2 families (paired with the restaged stage4_m2 stages); h_sonnet.json, h_opus.json: the "
    "Claude Sonnet 5.5 and Claude Opus 5.5 definition writers (H-a, H-c: H1, H2); h_b.json: the "
    "thinking Qwen3.5-9B writer at a 32,768-token limit (H-b: H3, not judged by its registered "
    "validity rule); g_a.json: block G on the G-a re-runs (G1, G2); g_b.json: block G on the 18 "
    "G-b program sets (G3); v.json: block V on its nine program sets, with the G-b sets of the "
    "same writers and samples for its per-writer comparison. The G-a, V and the Qwen3.5-9B, "
    "Claude Sonnet 5.5 and Claude Opus 5.5 G-b runs are re-runs from the response cache, and "
    "stage4_c_e1 and the stage4_m2 Llama-3.1-8B and gpt-oss-20b stages are restaged with re-runs "
    "(prereg/decisions.md 2026-10-17). u_delta.json is exploratory (analysis.ex_round3_upsert): "
    "where the upsert can act. x.json is block X (Mem0) on worlds 1000-1009: void, its anchor "
    "answers differ from E5's, so X1 is not judged and its cells are exploratory "
    "(prereg/decisions.md 2026-10-17)"
)
CLASSES = (("definition", "Definition"), ("other", "Other"), ("all", "All"))
U_ARMS = (("D0", "Dzero"), ("D1", "Done"), ("Dm", "Dm"))
Y_ARMS = (("D1", "Done"), ("Dm", "Dm"))
SR_ARMS = (("D1", "Main"), ("D1s", "Dm"))  # the store/reader split's arm keys
SR_CELLS = (
    ("D0B0", "DzeroBzero"),
    ("D0B1", "DzeroBone"),
    ("D1B0", "DoneBzero"),
    ("D1B1", "DoneBone"),
)
SR_TERMS = (("r", "R"), ("r_store", "RStore"), ("a", "A"), ("a_store", "AStore"))
SR_PARTS = (("interaction", "BxD"), ("store", "StorePart"), ("reader", "ReaderPart"))
PREDICTIONS = (
    ("u", "U1", "UOne"),
    ("u", "U3", "UThree"),
    ("u", "U4", "UFour"),
    ("u", "U5", "UFive"),
    ("n", "N1", "NOne"),
    ("y", "Y1", "YOne"),
    ("y", "Y2", "YTwo"),
    ("y", "Y3", "YThree"),
    ("q", "Q1", "QOne"),
    ("q", "Q2", "QTwo"),
    ("g_a", "G1", "GOne"),
    ("g_a", "G2", "GTwo"),
    ("g_b", "G3", "GThree"),  # G1 and G2 are judged in g_a.json, G3 in g_b.json
    ("h_b", "H3", "HThree"),
    ("v", "V1", "VOne"),
    ("v", "V2", "VTwo"),
    ("x", "X1", "XOne"),
)
VERDICT_WORDS = {
    "holds": "holds",
    "fails": "fails",
    "judged": "judged",  # a reading, not a pass or fail (U3, H1)
    "not_judged": "not judged",
}
READING_WORDS = {  # the readings of U3 and H1 as the paper words them
    "executability": "executability",
    "key_schedule": "key schedule",
    "both": "both",
    "writer_property": "writer property",
    "execution_pays": "execution pays",
}
H_ARMS = (("D1", "Done"), ("Dm", "Dm"))
DELTA_CLASSES = (("definition", "Definition"), ("other", "Other"))
DELTA_PARTS = (("yielded", "Yielded"), ("none", "None"))
X_ARMS = (("mem0_det", "Det"), ("mem0_rep", "Rep"))  # source-keyed invalidation, query-time view


# ------------------------------------------------------------------------------------ loading
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged results (None where the stage or a file is not there), and the committed
    reader's store/reader split (stage `ex_round3`)."""
    src = dict(sources)
    out: dict[str, Any] = {k: stage_json(root, src, STAGE_KEY, name) for k, name in FILES.items()}
    out["sr"] = stage_json(root, src, "ex_round3", "store_reader.json")
    return out


# ------------------------------------------------------------------------------------ formats
class _Out(dict[str, str]):
    """Macros of this block (prefix RThree): a name defined twice is an error."""

    def put(self, *parts: str, value: str) -> None:
        name = "".join(("RThree", *parts))
        if name in self:
            raise ValueError(f"macro {name} defined twice")
        self[name] = value


def ratio(x: float | None) -> str:
    """A ratio with two decimals (signed only when negative)."""
    if x is None or x != x:
        return MISSING
    v = round(x, 2)
    return ("$-$" if v < 0 else "") + f"{abs(v):.2f}"


def _est(parent: Mapping[str, Any] | None, key: str, f: Callable[[float | None], str] = pct) -> str:
    """An estimate with its interval under `parent`: MISSING when the parent is not there, a dash
    when the parent is there but the estimate is not."""
    if parent is None:
        return MISSING
    e = parent.get(key)
    return UNDEFINED if e is None else fmt(e, f)


def _count(x: Any) -> str:
    return MISSING if x is None else f"{int(x):,}"


def _verdict(res: Mapping[str, Any] | None, name: str) -> str:
    if res is None:
        return MISSING
    p = _get(res, "predictions", name)
    return MISSING if p is None else VERDICT_WORDS[str(p["verdict"])]


# ------------------------------------------------------------------------------------ macros
def _u_numbers(u: Mapping[str, Any] | None, out: _Out) -> None:
    fam = _get(u, "quantities", "families", "main")
    for arm, at in U_ARMS:
        for b, bt in BRANCH_TAGS:
            for cls, ct in CLASSES:
                q = None if u is None else _get(fam, "R", "refresh", b, arm, cls) or {}
                r = None if q is None else q.get("R")
                r_text = MISSING if q is None else fmt(r, ratio) if r else UNDEFINED
                out.put("U", at, bt, ct, "R", value=r_text)
                lev = None if q is None else q.get("update") or {}
                for key, kt in (("B0", "Bzero"), ("x", "Refresh"), ("B1", "Bone")):
                    out.put("U", at, bt, ct, "Update", kt, value=_est(lev, key))
                out.put("U", at, bt, ct, "Gain", value=_est(q, "numerator", pp))
                out.put("U", at, bt, ct, "BEffect", value=_est(q, "denominator", pp))
                out.put(
                    "U",
                    at,
                    bt,
                    ct,
                    "Items",
                    value=_count(None if q is None else q.get("n_items", 0)),
                )
                # the upsert cell on the same item-times (its B=0 and B=1 levels are those above)
                q = None if u is None else _get(fam, "R", "upsert", b, arm, cls) or {}
                r = None if q is None else q.get("R")
                out.put(
                    "U", at, bt, ct, "UpsertR", value=MISSING if q is None else _ratio_or_dash(r)
                )
                lev = None if q is None else q.get("update") or {}
                out.put("U", at, bt, ct, "UpdateUpsert", value=_est(lev, "x"))
                out.put("U", at, bt, ct, "UpsertGain", value=_est(q, "numerator", pp))
    for kind, kt in (("refresh", ""), ("upsert", "Upsert")):
        worlds = _get(fam, "worlds", kind)
        out.put("U", kt, "Worlds", value=MISSING if worlds is None else str(len(worlds)))
    tests = _get(u, "predictions", "U1", "tests")
    true = None if tests is None else sum(t["ok"] is True for t in tests)
    out.put("UOneTrue", value=_count(true))
    out.put("UAnchorCompared", value=_count(_get(u, "pairing", "anchor_answers_compared")))
    # U3: the pooled ratio of the upsert cells (D=1, D_m; S_src, S_def; definition items)
    pooled = None if u is None else _get(fam, "pooled_definition_D1_Dm_src_def", "upsert") or {}
    out.put(
        "UThreeR",
        value=MISSING if pooled is None else _ratio_or_dash(pooled.get("R")),
    )
    lev = None if pooled is None else pooled.get("update") or {}
    for key, kt in (("B0", "Bzero"), ("x", "Upsert"), ("B1", "Bone")):
        out.put("UThreeUpdate", kt, value=_est(lev, key))
    out.put("UThreeGain", value=_est(pooled, "numerator", pp))
    out.put("UThreeBEffect", value=_est(pooled, "denominator", pp))
    out.put("UThreeItems", value=_count(None if pooled is None else pooled.get("n_items", 0)))
    reading = _get(u, "predictions", "U3", "reading")
    out.put(
        "UThreeReading",
        value=MISSING if u is None else READING_WORDS[reading] if reading else UNDEFINED,
    )
    # U4: Update(D1B0_upsert) - Update(D1B0) after S_del, all items
    u4 = None if u is None else _get(fam, "U4") or {}
    out.put("UFourDiff", value=_est(u4, "difference", pp))
    out.put("UFourUpdateUpsert", value=_est(u4, "update_upsert"))
    out.put("UFourUpdateBzero", value=_est(u4, "update_B0"))
    diff = None if u4 is None else u4.get("difference") or {}
    out.put("UFourItems", value=_count(None if diff is None else diff.get("n_items", 0)))


def _ratio_or_dash(e: Mapping[str, Any] | None) -> str:
    """A ratio with its interval, or a dash when it is not evaluated."""
    return fmt(dict(e), ratio) if e else UNDEFINED


def _delta_numbers(d: Mapping[str, Any] | None, out: _Out) -> None:
    """Exploratory (`ex_round3_upsert`): where the upsert can act, per arm, branch, class and
    part (the delta yielded a new item for the slot since the fork, or not)."""
    arms = _get(d, "families", "main", "arms")
    for arm, at in U_ARMS:
        moved: dict[str, int] = {}
        for b, bt in BRANCH_TAGS:
            for cls, ct in DELTA_CLASSES:
                node = None if d is None else _get(arms, arm, b, cls) or {}
                effect = None if node is None else node.get("B_effect")
                for part, ptag in DELTA_PARTS:
                    p = None if node is None else _get(node, "parts", part) or {}
                    head = ("UDelta", at, bt, ct, ptag)
                    out.put(
                        *head, "Items", value=_count(None if p is None else p.get("n_items", 0))
                    )
                    out.put(*head, "Share", value=_share(None if p is None else p.get("share"), p))
                    lev = None if p is None else p.get("update") or {}
                    for key, kt in (("B0", "Bzero"), ("upsert", "Upsert"), ("B1", "Bone")):
                        out.put(*head, "Update", kt, value=_est(lev, key))
                    c = None if p is None else p.get("contribution_to_B_effect")
                    out.put(
                        *head,
                        "EffectShare",
                        value=_share(None if c is None or not effect else c / effect, p),
                    )
                    moves = None if p is None else p.get("upsert_differs") or {}
                    for key, kt in (("above_B0", "Above"), ("below_B0", "Below")):
                        n = None if moves is None else moves.get(key, 0)
                        out.put(*head, kt, value=_count(n))
                        if n is not None:
                            moved[cls] = moved.get(cls, 0) + int(n)
        for cls, ct in DELTA_CLASSES:  # answers the upsert moved, over the three shifts
            out.put("UDelta", at, "Shifts", ct, "Moved", value=_count(moved.get(cls)))


def _share(x: float | None, parent: Mapping[str, Any] | None) -> str:
    """A share in percent without decimals: MISSING when the parent is not there, a dash when the
    share is not defined."""
    if parent is None:
        return MISSING
    return UNDEFINED if x is None else pct(x, 0)


def _h_numbers(inp: Mapping[str, Any], out: _Out) -> None:
    """Block H per definition-item writer (`H_WRITERS`)."""
    for h in H_WRITERS:
        res = inp.get(h.key)
        q = _get(res, "quantities", "writers", h.model)
        tag = ("H", h.tag)
        for pred, pt in (("H1", "One"), ("H2", "Two")):
            out.put(*tag, pt, "Verdict", value=_verdict(res, pred + h.suffix))
        reading = _get(res, "predictions", "H1" + h.suffix, "reading")
        out.put(
            *tag,
            "OneReading",
            value=MISSING if res is None else READING_WORDS[reading] if reading else UNDEFINED,
        )
        for b, bt in BRANCH_TAGS:
            for cls, ct in CLASSES:
                node = None if res is None else _get(q, "bxd", b, cls) or {}
                for arm, at in H_ARMS:
                    out.put(*tag, bt, ct, at, "BxD", value=_est(node, arm, pp))
        r0 = None if res is None else _get(q, "r0") or {}
        out.put(*tag, "RZero", value=_est(r0, "writer"))
        out.put(*tag, "RZeroQwen", value=_est(r0, "qwen_same_worlds"))
        out.put(*tag, "RZeroDiff", value=_est(r0, "paired_difference", pp))
        reg = None if r0 is None else r0.get("registered_qwen")
        out.put(*tag, "RZeroRegistered", value=MISSING if reg is None else pct(reg))
        out.put(*tag, "MissedZero", value=_est(r0, "writer_m0"))
        v = _get(q, "validity")
        out.put(*tag, "Writes", value=_count(_get(v, "calls", "distinct_write_requests")))
        out.put(*tag, "AtLimit", value=_count(_get(v, "calls", "at_limit")))
        out.put(*tag, "Unreadable", value=_count(_get(v, "cache", "unreadable")))
        out.put(*tag, "CellWrites", value=_count(_get(v, "cell_writes", "writes")))
        tokens = _get(v, "calls", "completion_tokens_mean")
        out.put(*tag, "TokensMean", value=MISSING if tokens is None else f"{tokens:,.0f}")
        out.put(*tag, "TokensMax", value=_count(_get(v, "calls", "completion_tokens_max")))
        worlds = _get(q, "worlds")
        out.put(*tag, "Worlds", value=MISSING if worlds is None else str(len(worlds)))
        out.put(
            *tag, "AnchorCompared", value=_count(_get(res, "pairing", "anchor_answers_compared"))
        )


def _q_numbers(q: Mapping[str, Any] | None, out: _Out) -> None:
    """Block Q (S_def, definition items): Q1's B effect at D=1 with probes that do not name the
    definition version and with the committed probes, their paired difference, and B x D with
    each per arm (Q2 is judged at D=1; D_m is reported)."""
    q1 = None if q is None else _get(q, "quantities", "Q1") or {}
    for key, kt in (
        ("B_effect_D1_unnamed", "BEffectUnnamed"),
        ("B_effect_D1_named", "BEffectNamed"),
        ("difference", "OneDiff"),
    ):
        out.put("Q", kt, value=_est(q1, key, pp))
    diff = None if q1 is None else q1.get("difference") or {}
    out.put("QItems", value=_count(None if diff is None else diff.get("n_items", 0)))
    for arm, at in Y_ARMS:
        node = None if q is None else _get(q, "quantities", "Q2", arm) or {}
        for key, kt in (("unnamed", "Unnamed"), ("named", "Named")):
            out.put("Q", at, "BxD", kt, value=_est(node, key, pp))
    worlds = _get(q, "quantities", "worlds")
    out.put("QWorlds", value=MISSING if worlds is None else str(len(worlds)))
    out.put("QAnchorCompared", value=_count(_get(q, "pairing", "anchor_answers_compared")))


def _ga_numbers(g: Mapping[str, Any] | None, out: _Out) -> None:
    """Block G on the held-out-definition runs: B x D on definition items per branch and arm,
    with the two-way interval (worlds x definitions) and the one-way world interval."""
    held = _get(g, "quantities", "heldout")
    for b, bt in BRANCH_TAGS:
        for arm, at in H_ARMS:
            node = None if g is None else _get(held, b, arm) or {}
            head = ("GA", bt, at)
            out.put(*head, "BxD", value=_est(node, "two_way", pp))
            out.put(*head, "BxDOneWay", value=_est(node, "one_way_worlds", pp))
            two = None if node is None else node.get("two_way") or {}
            out.put(*head, "Items", value=_count(None if two is None else two.get("n_items", 0)))
            defs = None if node is None else node.get("definitions_with_items") or []
            out.put(*head, "Definitions", value=_count(None if defs is None else len(defs)))
            per = None if node is None else node.get("per_definition") or {}
            above = None if per is None else sum(e["est"] > 0 for e in per.values())
            out.put(*head, "Positive", value=_count(above))
    draws = _get(g, "quantities", "heldout_draws")
    out.put("GADraws", value=MISSING if draws is None else str(len(draws)))
    out.put("GAWorlds", value=_count(_get(held, "def", "D1", "two_way", "n_worlds")))


def _gb_numbers(g: Mapping[str, Any] | None, out: _Out) -> None:
    """Block G-b (G3): B x D at D_m on definition items over the 18 program sets of six writers,
    per branch (two-way and one-way world intervals) and per writer (world bootstrap over the
    writer's sets); the writer share of the variance of the per-set B x D after S_def."""
    q = _get(g, "quantities", "programs")
    prog = _get(q, "branches")
    for b, bt in BRANCH_TAGS:
        node = None if g is None else _get(prog, b, "definition") or {}
        out.put("GB", bt, "BxD", value=_est(node, "two_way", pp))
        out.put("GB", bt, "BxDOneWay", value=_est(node, "one_way_worlds", pp))
        two = None if node is None else node.get("two_way") or {}
        out.put("GB", bt, "Items", value=_count(None if two is None else two.get("n_items", 0)))
        sets = None if node is None else node.get("sets_with_items") or []
        out.put("GB", bt, "Sets", value=_count(None if sets is None else len(sets)))
        for model, wt in G_B_WRITERS:
            per = None if node is None else node.get("per_writer") or {}
            out.put("GB", wt, bt, "BxD", value=_est(per, model, pp))
    # eta squared: between-writer over total sum of squares of the 18 per-set B x D at D_m after
    # S_def on definition items; the one-way random-effects share of the same values beside it
    share = None if g is None else _get(q, "writer_variance_share") or {}
    for key, kt in (("eta_squared", "EtaSq"), ("variance_component_share", "VarShare")):
        x = None if share is None else share.get(key)
        out.put(
            "GBDefWriter",
            kt,
            value=MISSING if share is None else UNDEFINED if x is None else ratio(x),
        )
    out.put("GBWriters", value=_count(None if share is None else share.get("writers")))
    worlds = _get(prog, "def", "definition", "two_way", "n_worlds")
    out.put("GBWorlds", value=_count(worlds))
    out.put("GBAnchorCompared", value=_count(_get(g, "pairing", "anchor_answers_compared")))


def _think_numbers(h: Mapping[str, Any] | None, out: _Out) -> None:
    """Block H-b (H3): the validity of the thinking Qwen3.5-9B writer at its token limit, which
    decides whether H3 is judged: thinking writes, those cut off at the limit and their share,
    the unreadable ones and their share."""
    v = _get(h, "quantities", "think", "validity")
    limits = None if v is None else v.get("limits") or []
    out.put("HThinkLimit", value=MISSING if limits is None else ", ".join(f"{x:,}" for x in limits))
    for key, kt in (("writes", "Writes"), ("cut_off", "CutOff"), ("unreadable", "Unreadable")):
        out.put("HThink", kt, value=_count(None if v is None else v.get(key)))
    for key, kt in (("cut_off_share", "CutOffShare"), ("unreadable_share", "UnreadableShare")):
        x = None if v is None else v.get(key)
        out.put("HThink", kt, value=MISSING if v is None else pct(x))
    worlds = _get(h, "quantities", "think", "worlds")
    out.put("HThinkWorlds", value=MISSING if worlds is None else str(len(worlds)))
    out.put("HThinkAnchorCompared", value=_count(_get(h, "pairing", "anchor_answers_compared")))


def _u5_numbers(u: Mapping[str, Any] | None, out: _Out) -> None:
    """U5: per M2 family U3's pooled upsert ratio over D=1 and S_src, S_def on definition items,
    its reading, and how many families give the main writer's U3 reading."""
    fam = _get(u, "quantities", "families")
    u5 = _get(u, "predictions", "U5")
    per = None if u5 is None else u5.get("families") or {}
    main = None if u5 is None else u5.get("main_reading")
    same = 0
    for f in FAMILIES:
        pooled = None if u is None else _get(fam, f.model, "pooled_definition_D1_Dm_src_def")
        node = None if pooled is None else pooled.get("upsert") or {}
        out.put(
            "UFive", f.tag, "R", value=MISSING if node is None else _ratio_or_dash(node.get("R"))
        )
        out.put(
            "UFive", f.tag, "Items", value=_count(None if node is None else node.get("n_items"))
        )
        reading = None if per is None else _get(per, f.model, "reading")
        out.put(
            "UFive",
            f.tag,
            "Reading",
            value=MISSING if per is None else READING_WORDS[reading] if reading else UNDEFINED,
        )
        same += bool(reading) and reading == main
    out.put("UFiveSame", value=MISSING if per is None else str(same))
    out.put("UFiveFamilies", value=MISSING if per is None else str(len(per)))


def _v_numbers(v: Mapping[str, Any] | None, out: _Out) -> None:
    """Block V: B x D at D_m on definition items over the nine program sets, per branch (two-way
    and one-way world intervals) and per writer; per writer, V minus G-b with V's level."""
    q = _get(v, "quantities")
    prog = _get(q, "programs", "branches")
    for b, bt in BRANCH_TAGS:
        node = None if v is None else _get(prog, b, "definition") or {}
        out.put("V", bt, "BxD", value=_est(node, "two_way", pp))
        out.put("V", bt, "BxDOneWay", value=_est(node, "one_way_worlds", pp))
        two = None if node is None else node.get("two_way") or {}
        out.put("V", bt, "Items", value=_count(None if two is None else two.get("n_items", 0)))
        sets = None if node is None else node.get("sets_with_items") or []
        out.put("V", bt, "Sets", value=_count(None if sets is None else len(sets)))
        for model, wt in V_WRITERS:
            per = None if node is None else node.get("per_writer") or {}
            out.put("V", wt, bt, "BxD", value=_est(per, model, pp))
    for model, wt in V_WRITERS:
        node = None if v is None else _get(q, "V_minus_G_b", model) or {}
        upd = None if node is None else node.get("update_D1sB1_source") or {}
        for group, gt in (*BRANCH_TAGS, ("pooled", "Pooled")):
            e = None if upd is None else upd.get(group) or {}
            out.put("V", wt, "Update", gt, value=_est(e, "V"))
            out.put("V", wt, "Update", gt, "Diff", value=_est(e, "difference", pp))
        agree = None if node is None else node.get("agreement_t0") or {}
        out.put("V", wt, "Agree", value=_est(agree, "V"))
        out.put("V", wt, "Agree", "Diff", value=_est(agree, "difference", pp))
    out.put("VAnchorCompared", value=_count(_get(v, "pairing", "anchor_answers_compared")))


def _x_numbers(x: Mapping[str, Any] | None, out: _Out) -> None:
    """Block X, void (every value exploratory): per branch and class the changed item-times,
    native Mem0's m, r and a (it only adds, so m = 1 and r is undefined), the invalidation arm's
    m and r, and each arm's effect on Update; the worlds and the store mutations after the fork.
    The anchor counts that void the run stay in x.json: the paper does not state them."""
    cells = _get(x, "quantities", "cells")
    for b, bt in BRANCH_TAGS:
        for cls, ct in CLASSES[:2]:
            cell = None if x is None else _get(cells, f"{b}/{cls}") or {}
            arms = None if cell is None else cell.get("arms") or {}
            native = None if arms is None else arms.get("mem0") or {}
            det = None if arms is None else arms.get("mem0_det") or {}
            effects = None if cell is None else cell.get("effects") or {}
            head = ("X", bt, ct)
            out.put(*head, "Items", value=_count(None if native is None else native.get("n", 0)))
            for key, kt in (("m", "M"), ("r", "R"), ("a", "A")):
                out.put(*head, kt, value=_est(native, key))
            for key, kt in (("m", "M"), ("r", "R")):
                out.put(*head, "Det", kt, value=_est(det, key))
            for arm, at in X_ARMS:
                out.put(*head, at, "Effect", value=_est(effects, arm, pp))
    sources = _get(x, "pairing", "detail", "prefix_sources")
    out.put("XWorlds", value=_count(None if sources is None else len(sources)))
    mutations = _get(x, "reported", "mutations_after_fork")
    for arm, at in (("mem0", "Native"), ("mem0_det", "Det")):
        node = None if x is None else (mutations or {}).get(arm) or {}
        out.put("X", at, "Mutations", value=_count(None if node is None else sum(node.values())))


def _n_numbers(n: Mapping[str, Any] | None, out: _Out) -> None:
    cf = _get(n, "quantities", "counterfactual")
    moved = _get(n, "reported", "natural_also_changed")
    for b, bt in BRANCH_TAGS:
        for cls, ct in CLASSES:
            node = None if n is None else _get(cf, b, "shift", cls) or {}
            out.put("N", bt, ct, "Shifted", value=_est(node, "shifted_branch", pp))
            out.put("N", bt, ct, "Natural", value=_est(node, "natural", pp))
            out.put("N", bt, ct, "Diff", value=_est(node, "difference", pp))
            out.put(
                "N",
                bt,
                ct,
                "Items",
                value=_count(None if node is None else node.get("item_times_in_branch", 0)),
            )
            worlds = _get(node, "difference", "n_worlds")
            out.put("N", bt, ct, "Worlds", value=_count(worlds) if node is not None else MISSING)
            mv = None if n is None else _get(moved, b) or {}
            out.put("N", bt, ct, "NaturalMoved", value=_est(mv, cls))
    out.put("NAnchorCompared", value=_count(_get(n, "pairing", "anchor_answers_compared")))


def _y_numbers(y: Mapping[str, Any] | None, out: _Out) -> None:
    con = _get(y, "quantities", "contrasts")
    for b, bt in BRANCH_TAGS:
        for cls, ct in CLASSES:
            for arm, at in Y_ARMS:
                node = None if y is None else _get(con, b, cls, arm) or {}
                out.put("Y", bt, ct, at, "BxD", value=_est(node, "reader", pp))
                out.put("Y", bt, ct, at, "Committed", value=_est(node, "committed_reader", pp))
                out.put("Y", bt, ct, at, "Diff", value=_est(node, "difference", pp))
                out.put(
                    "Y",
                    bt,
                    ct,
                    at,
                    "Items",
                    value=_count(None if node is None else node.get("n_items", 0)),
                )
    v = _get(y, "quantities", "validity")
    for key, kt in (
        ("answers", "Answers"),
        ("unreadable", "Unreadable"),
        ("cut_off", "CutOff"),
        ("reader_calls", "Calls"),
    ):
        out.put("Y", kt, value=_count(_get(v, key)))
    out.put("YContextCompared", value=_count(_get(y, "pairing", "anchor_answers_compared")))
    out.put(
        "YContextDiffers",
        value=_count(
            None if y is None else _get(y, "pairing", "counts", "context_hash_differs") or 0
        ),
    )
    cells = _get(y, "pairing", "detail", "context_differs", "by_cell")
    pairs = None if y is None else {tuple(k.split("|")[1:]) for k in (cells or {})}
    out.put("YContextWorldBranches", value=_count(None if pairs is None else len(pairs)))
    out.put(
        "YStoreRequestsSent",
        value=_count(
            None if y is None else _get(y, "pairing", "counts", "store_requests_sent") or 0
        ),
    )
    sr = _get(y, "quantities", "store_reader", "results")
    lost = _get(y, "quantities", "store_reader", "lost_at_D1B1")
    for arm, at in SR_ARMS:
        for b, bt in BRANCH_TAGS:
            for cls, ct in CLASSES:
                node = None if y is None else _get(sr, arm, b, cls) or {}
                inter = None if node is None else node.get("interaction") or {}
                for key, kt in SR_PARTS:
                    out.put("YSR", at, bt, ct, kt, value=_est(inter, key, pp))
                for cell, cellt in SR_CELLS:
                    terms = None if node is None else _get(node, "cells", cell) or {}
                    for key, kt in SR_TERMS:
                        out.put("YSR", at, bt, ct, cellt, kt, value=_est(terms, key))
        for cls, ct in CLASSES:
            tot = None if y is None else _get(lost, arm, cls, "shifts") or {}
            for key, kt in (
                ("right_stores", "RightStores"),
                ("lost", "Lost"),
                ("right_abstain", "RightAbstain"),
                ("lost_abstain", "LostAbstain"),
            ):
                out.put(
                    "YSR",
                    at,
                    "Shifts",
                    ct,
                    kt,
                    value=_count(None if tot is None else tot.get(key, 0)),
                )


def numbers(inp: Mapping[str, Any]) -> dict[str, str]:
    """Every macro of this block, from `load`."""
    out = _Out()
    for key, name, tag in PREDICTIONS:
        out.put(tag, "Verdict", value=_verdict(inp.get(key), name))
    u = inp.get("u")
    u2 = None if u is None else _get(u, "reported", "U2_R_refresh_D0_definition")
    out.put("UTwoVerdict", value=MISSING if u is None else "reported")
    for b, bt in (("src", "Src"), ("def", "Def")):
        node = None if u is None else _get(u2, b) or {}
        r = None if node is None else node.get("R")
        out.put("UTwo", bt, value=MISSING if node is None else fmt(r, ratio) if r else UNDEFINED)
    _u_numbers(u, out)
    _u5_numbers(u, out)
    _delta_numbers(inp.get("u_delta"), out)
    _n_numbers(inp.get("n"), out)
    _y_numbers(inp.get("y"), out)
    _h_numbers(inp, out)
    _q_numbers(inp.get("q"), out)
    _ga_numbers(inp.get("g_a"), out)
    _gb_numbers(inp.get("g_b"), out)
    _think_numbers(inp.get("h_b"), out)
    _v_numbers(inp.get("v"), out)
    _x_numbers(inp.get("x"), out)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ------------------------------------------------------------------------------------ tables
def _ratio_iv(e: Mapping[str, Any] | None) -> str:
    if not e:
        return UNDEFINED
    return interval(ratio(e["est"]), ratio(e["lo"]), ratio(e["hi"]))


def _pc(e: Mapping[str, Any] | None) -> str:
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=1)


def _vmark(res: Mapping[str, Any] | None, name: str) -> str:
    v = _verdict(res, name)
    return {"holds": r"\pass{}", "fails": r"\fail{}"}.get(v, v)


def _r0_iv(e: Mapping[str, Any] | None) -> str:
    """A share in percent with its interval, in the table format."""
    if not e:
        return MISSING
    return interval(_num(e["est"]) + r"\%", _num(e["lo"]), _num(e["hi"]))


def _refresh_cell(inp: Mapping[str, Any], arm: str, b: str, kind: str = "refresh") -> Any:
    """The refresh (or upsert) node of the main writer's family for one arm and shift."""
    node = _get(inp.get("u"), "quantities", "families", "main", "R", kind, b, arm, "definition")
    return None if node is None else dict(node)


def tab_refresh_upsert(inp: Mapping[str, Any]) -> str:
    """Appendix: Update of the refresh and upsert cells beside the append-only (B=0) and bound
    (B=1) cells on changed definition item-times, with the share R of binding's effect each
    recovers (the levels behind the registered refresh and upsert predictions)."""
    u = inp.get("u")

    def r_of(arm: str, b: str, kind: str = "refresh") -> str:
        if u is None:
            return MISSING
        q = _refresh_cell(inp, arm, b, kind)
        return _ratio_iv(None if q is None else q.get("R"))

    lines = [
        r"\begin{table}[t]",
        r"\caption{Refresh and upsert against binding (20 worlds, changed definition "
        r"item-times, source-epoch key, Update in percent). $R$: the share of binding's effect "
        r"on Update that a $B{=}0$ variant recovers, $(x-B_0)/(B_1-B_0)$, with its 95\% "
        r"world-bootstrap interval, not evaluated (" + UNDEFINED + r") when $B_1-B_0<5$\,pp.}",
        r"\label{tab:round3}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll rrrr cc@{}}",
        r"\toprule",
        r"& & $B{=}0$ & refresh & upsert & $B{=}1$ & $R$, refresh & $R$, upsert \\",
        r"\midrule",
    ]
    arms = ((r"$D{=}0$", "D0"), (r"$D{=}1$", "D1"), (r"$D_{\mathrm{m}}$", "Dm"))
    for i, (label, arm) in enumerate(arms):
        for k, (b, _) in enumerate(BRANCH_TAGS):
            lev = _get(_refresh_cell(inp, arm, b), "update")
            ups = _get(_refresh_cell(inp, arm, b, "upsert"), "update", "x")
            lines.append(
                _row(
                    label if k == 0 else "",
                    BRANCH_LABELS[b],
                    *(_pc(_get(lev, key)) if u else MISSING for key in ("B0", "x")),
                    _pc(ups) if u else MISSING,
                    _pc(_get(lev, "B1")) if u else MISSING,
                    r_of(arm, b),
                    r_of(arm, b, "upsert"),
                )
            )
        if i < len(arms) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_natural(inp: Mapping[str, Any]) -> str:
    """Appendix: the natural-branch test by shift and class, B x D on the item-times the shift
    changed, on the shifted branch and on the natural branch (same events up to the fork, no
    shift), their paired difference and the share of these item-times that also moved without
    the shift."""
    n = inp.get("n")
    cf = _get(n, "quantities", "counterfactual")
    lines = [
        r"\begin{table}[t]",
        r"\caption{The interaction with and without the shift (20 worlds, source-epoch key). "
        r"Shifted: $B{\times}D$ on the item-times the shift changed. Natural: $B{\times}D$ on "
        r"the same item-times of the natural branch, which has the same events up to the fork "
        r"and no shift. Difference: shifted minus natural, paired. Moved: share of these "
        r"item-times whose value also changed on the natural branch. pp, 95\% world-bootstrap "
        r"intervals. $\Sdef$ changes only definition items.}",
        r"\label{tab:natural}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll r ccc r@{}}",
        r"\toprule",
        r"& & $n$ & shifted & natural & difference & moved (\%) \\",
        r"\midrule",
    ]
    moved = _get(n, "reported", "natural_also_changed")
    for i, (b, _) in enumerate(BRANCH_TAGS):
        for k, (cls, _) in enumerate(CLASSES[:2]):
            node = _get(cf, b, "shift", cls)
            if n is not None and not (node or {}).get("item_times_in_branch"):
                continue  # a class the shift changes on no item-time (S_def: the other items)
            if n is None:
                vals = [MISSING] * 5
            else:
                vals = [
                    _count((node or {}).get("item_times_in_branch", 0)),
                    *(
                        _cell(_get(node, key))
                        for key in ("shifted_branch", "natural", "difference")
                    ),
                    _pc(_get(moved, b, cls)),
                ]
            lines.append(_row(BRANCH_LABELS[b] if k == 0 else "", cls, *vals))
        if i < len(BRANCH_TAGS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_reader(inp: Mapping[str, Any]) -> str:
    """Appendix: B x D on Update with the committed reader (Qwen3.5-9B) and with Claude Sonnet
    5.5 on the same stores, by shift and class (authored D=1), with the store part (the same for
    both readers) and each reader's part."""
    y, sr = inp.get("y"), _get(inp.get("sr"), "results")
    con = _get(y, "quantities", "contrasts")
    ysr = _get(y, "quantities", "store_reader", "results")
    lines = [
        r"\begin{table}[t]",
        r"\caption{The $B{\times}D$ interaction on Update with two readers of the same stores "
        r"(authored $D{=}1$, source-epoch key, 20 worlds, pp, 95\% world-bootstrap intervals). "
        r"Qwen: Qwen3.5-9B, the main study's reader. Sonnet: Claude Sonnet 5.5 (registered). "
        r"Difference: paired per item-time. Store: the part of $B{\times}D$ due to the stored "
        r"value, the same for both readers. Reader: the rest (exploratory split).}",
        r"\label{tab:reader}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll ccc c cc@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{$B{\times}D$} & & \multicolumn{2}{c}{reader part} \\",
        r"\cmidrule(lr){3-5}\cmidrule(l){7-8}",
        r"& & Qwen & Sonnet & difference & store part & Qwen & Sonnet \\",
        r"\midrule",
    ]
    for i, (b, _) in enumerate(BRANCH_TAGS):
        for k, (cls, _) in enumerate(CLASSES[:2]):
            node = _get(con, b, cls, "D1")
            q = _get(sr, "D1", b, cls, "interaction")
            s = _get(ysr, "D1", b, cls, "interaction")
            vals = [
                *(
                    _cell(_get(node, key)) if y else MISSING
                    for key in ("committed_reader", "reader", "difference")
                ),
                _cell(_get(s, "store")) if y else MISSING,
                _cell(_get(q, "reader")) if sr else MISSING,
                _cell(_get(s, "reader")) if y else MISSING,
            ]
            lines.append(_row(BRANCH_LABELS[b] if k == 0 else "", cls, *vals))
        if i < len(BRANCH_TAGS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _rule(d: Any) -> str:
    """A held-out definition's v1 and what v2 changes (the window or the threshold)."""
    v1, v2 = d.v1, d.v2
    first = (
        rf"${{\ge}}${v1.threshold} in {v1.window}\,d"
        if v1.aggregate == "count"
        else rf"{v1.window}\,d"
    )
    second = rf"${{\ge}}${v2.threshold}" if d.changed == "threshold" else rf"{v2.window}\,d"
    return rf"{first} $\to$ {second}"


def tab_heldout(inp: Mapping[str, Any]) -> str:
    """Appendix: the held-out definitions of G-a (`gen.kappa.heldout_draw`, a function of the
    sampler seeds) with each definition's own B x D at the authored D=1 per branch."""
    from memshift.gen import kappa  # lazy: generates the ten rejection worlds once

    ga = inp.get("g_a")
    held = _get(ga, "quantities", "heldout")
    lines = [
        r"\begin{table}[t]",
        r"\caption{The 18 held-out definitions, three draws of six fixed before any model call. "
        r"Each definition admits the "
        r"records of one source that pass a predicate in a window. Count: the people or projects "
        r"with at least the threshold of admitted records. Members: the admitted tasks. Span: the "
        r"earliest start to the latest end of the admitted meetings, in whole hours. v2 changes "
        r"the window or the threshold. $B{\times}D$: the definition's own interaction on its "
        r"changed item-times, authored $D{=}1$, Qwen3.5-9B, 10 worlds (pp, point estimates, "
        r"exploratory). A per-item double difference lies between $-$200 and $+$200\,pp.}",
        r"\label{tab:heldout}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l l l l l rrr@{}}",
        r"\toprule",
        r"& & & & & \multicolumn{3}{c}{$B{\times}D$} \\",
        r"\cmidrule(l){6-8}",
        r"draw & definition & records & aggregate & v1 $\to$ v2 & $\Ssrc$ & $\Sdef$ & $\Sdel$ \\",
        r"\midrule",
    ]
    for i, seed in enumerate(G_A_SEEDS):
        for k, d in enumerate(kappa.heldout_draw(seed).defs):
            vals = []
            for b, _ in BRANCH_TAGS:
                e = _get(held, b, "D1", "per_definition", f"heldout:{seed}|k.{d.kappa_id}")
                vals.append(
                    MISSING if ga is None else UNDEFINED if e is None else _num(e["est"], sign=True)
                )
            lines.append(
                _row(
                    str(i + 1) if k == 0 else "",  # the draw's number, not its sampler seed
                    d.v1.form.name.replace("_", " "),
                    f"{d.v1.source}: {d.v1.predicate.replace('_', ' ')}",
                    d.v1.aggregate,
                    _rule(d),
                    *vals,
                )
            )
        if i < len(G_A_SEEDS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_userwords(inp: Mapping[str, Any]) -> str:
    """Appendix: block V per program writer: B x D at D_m on definition items per branch, and V
    minus G-b (the same writer's programs from the codified statement, same samples) in Update on
    changed definition items at D1sB1_source (three branches pooled) and in agreement at t0."""
    v = inp.get("v")
    q = _get(v, "quantities")
    prog = _get(q, "programs", "branches")
    names = {"Qwen": "Qwen3.5-9B", "Sonnet": "Claude Sonnet 5.5", "Opus": "Claude Opus 5.5"}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Definitions in the user's own words per program writer, three program sets "
        r"each, 20 worlds, definition items. $B{\times}D$: the interaction at "
        r"$D_{\mathrm{m}}$ (pp, point estimates). Update: changed definition item-times at "
        r"$D_{\mathrm{m}}$ with binding, the three shifts pooled. Agreement: the programs' values "
        r"before the shift that equal gold. Words: the programs from the user's words. Difference: "
        r"words minus the same writer's programs from the codified statement, same samples, paired "
        r"per item (pp, 95\% world-bootstrap intervals).}",
        r"\label{tab:userwords}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l rrr rc rc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{$B{\times}D$} & \multicolumn{2}{c}{Update (\%)} "
        r"& \multicolumn{2}{c}{agreement (\%)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
        r"writer & $\Ssrc$ & $\Sdef$ & $\Sdel$ & words & difference & words & difference \\",
        r"\midrule",
    ]
    for model, wt in V_WRITERS:
        node = _get(q, "V_minus_G_b", model)
        upd = _get(node, "update_D1sB1_source", "pooled")
        agree = _get(node, "agreement_t0")
        if v is None:
            vals = [MISSING] * 7
        else:
            bxd = [_get(prog, b, "definition", "per_writer", model) for b, _ in BRANCH_TAGS]
            vals = [
                *(_num(e["est"], sign=True) if e else UNDEFINED for e in bxd),
                _pc(_get(upd, "V")),
                _cell(_get(upd, "difference")),
                _pc(_get(agree, "V")),
                _cell(_get(agree, "difference")),
            ]
        lines.append(_row(names[wt], *vals))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_mem0x(inp: Mapping[str, Any]) -> str:
    """Appendix: block X per branch and class (every value exploratory; the caption does not
    discuss the void run): native Mem0's m, r and a, the invalidation arm's m and r, and each
    arm's effect on Update with its interval."""
    x = inp.get("x")
    cells = _get(x, "quantities", "cells")
    lines = [
        r"\begin{table}[t]",
        r"\caption{Interventions on Mem0 on ten three-month worlds, changed item-times. "
        r"The cells are exploratory. $m$, $r$, $a$: the terms of "
        r"$\mathrm{Update}=(1-m)\,r+m\,a$ (percent). Mem0 only adds memories, so it invalidates "
        r"nothing ($m{=}1$) and $r$ is undefined. Invalidation: before a record's revision or "
        r"deletion is ingested, the memories of its earlier revision are deleted. Query-time view: "
        r"the evidence for a definition item starts with the value of the writer's program for the "
        r"item, run on the live records. Other items are served as by Mem0. Effect: Update minus "
        r"Mem0's, paired per item-time (pp, 95\% world-bootstrap intervals).}",
        r"\label{tab:mem0x}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll r rrr rr c c@{}}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{Mem0} & \multicolumn{3}{c}{invalidation} & query-time view \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(l){10-10}",
        r"& & $n$ & $m$ & $r$ & $a$ & $m$ & $r$ & effect & effect \\",
        r"\midrule",
    ]
    for i, (b, _) in enumerate(BRANCH_TAGS):
        for k, (cls, _) in enumerate(CLASSES[:2]):
            cell = _get(cells, f"{b}/{cls}")
            native, det = _get(cell, "arms", "mem0"), _get(cell, "arms", "mem0_det")
            if x is None:
                vals = [MISSING] * 8
            else:
                vals = [
                    _count(_get(native, "n")),
                    *(_pc(_get(native, key)) for key in ("m", "r", "a")),
                    *(_pc(_get(det, key)) for key in ("m", "r")),
                    *(_cell(_get(cell, "effects", arm)) for arm, _ in X_ARMS),
                ]
            lines.append(_row(BRANCH_LABELS[b] if k == 0 else "", cls, *vals))
        if i < len(BRANCH_TAGS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tables(inp: Mapping[str, Any]) -> dict[str, str]:
    return {
        "tab_refresh_upsert.tex": tab_refresh_upsert(inp),
        "tab_natural.tex": tab_natural(inp),
        "tab_reader.tex": tab_reader(inp),
        "tab_heldout.tex": tab_heldout(inp),
        "tab_userwords.tex": tab_userwords(inp),
        "tab_mem0x.tex": tab_mem0x(inp),
    }


# ----------------------------------------------------------------------------------- staging
def source_runs() -> list[str]:
    """Every run the staged results read, once each, in the order of `FILES`, then the stages."""
    return list(dict.fromkeys([*(r for key in FILES for r in RUNS[key]), *COMPARE, *M2_COMPARE]))


def stage(root: Path, out: Path, draws: int | None = None) -> None:
    """Rerun every staged result from the raw runs under ROOT/runs into OUT/analysis and write the
    stage manifest (the commands of `PRODUCERS`, the source runs of `source_runs` and the SHA256
    of each response-cache snapshot). Paths are recorded relative to ROOT."""
    from memshift.analysis import ex_round3_upsert
    from memshift.analysis import round3_registered as rr
    from memshift.analysis.stage import write_stage_manifest
    from memshift.runinfo import file_sha256

    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    n_draws = draws or rr.DRAWS
    for key, name in FILES.items():
        runs = [root / r for r in RUNS[key]]
        compare = [root / c for c in _compare(key)]
        if key == "u_delta":
            res = ex_round3_upsert.analyse(runs, compare, n_draws)
        else:
            cache = root / CACHES[key] if key in CACHES else None
            res = rr.run_block(BLOCKS[key], runs, compare, cache, n_draws)
        res["inputs"] = {"runs": list(RUNS[key]), "compare": list(_compare(key))}
        if key in CACHES:
            res["inputs"]["response_cache"] = CACHES[key]
        (analysis / name).write_text(json.dumps(res, indent=1, default=str) + "\n")
    extra = {
        "producers": PRODUCERS,
        "note": NOTE,
        "response_caches": {c: file_sha256(root / c) for c in CACHES.values()},
    }
    write_stage_manifest(out, STAGE_KEY, [root / r for r in source_runs()], extra)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Stage the registered round-3 results (docstring).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None, help=f"default: ROOT/{STAGE_DIR}")
    a = ap.parse_args(argv)
    stage(a.root, a.out or a.root / STAGE_DIR)


if __name__ == "__main__":
    main()
