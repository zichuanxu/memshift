"""The memory systems on the (D,B) grid as a LaTeX table for the paper appendix: the sixteen-system
audit (audit/system_audit.json), the six native systems of the native-system experiment and our
own cells, generated so that the paper never diverges from the audit record or the experiment.

Rows: the sixteen audited systems plus our cells (group != 'extra (starting audit)'; the extra rows
of the starting audit stay in the audit record only) and the native systems that the audit does not
cover, grouped as products, memory libraries, research systems and data stores, and our baselines
and cells. An audited system takes its audit placement; a native system that the audit does not
cover takes the label the native-system experiment reads from its documentation and code
(`analysis.e5.DOCUMENTED_LABELS`). A native system whose audit placement differs from that label
gets a note with the label the experiment measured (the `\\Nat<tag>Label` macro). The basis column
gives what a placement rests on (code, paper, documentation, indirect evidence) and, for a system
run natively, the property of its code that sets its B label. The reasons and evidence of every
placement are in the audit record; the versions of the native systems are pinned in the released
configuration (docker/requirements.lock, docker/native/*.txt), not printed.

Usage: python -m memshift.analysis.audit_table > paper/generated/audit_table.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

from memshift.analysis.e5 import DOCUMENTED_LABELS
from memshift.analysis.paper_data import NATIVES

EXTRA = "extra (starting audit)"


class NativeRow(NamedTuple):
    """A system of the native-system experiment: its key there, its short name in the audit (None
    when the audit does not cover it), its citation and the property of its code that sets B."""

    key: str
    audit_name: str | None
    cite: str
    reason: str


NATIVE_ROWS = (
    NativeRow(
        "mem0",
        "Mem0 (self-hosted)",
        "arxiv2504_19413",
        "the add path with inference emits only new memories, and updates and deletions need"
        " explicit calls",
    ),
    NativeRow(
        "amem",
        None,
        "xu2025amem",
        "every event becomes a new note, and nothing supersedes or deletes a note",
    ),
    NativeRow(
        "memos",
        None,
        "li2025memos",
        r"\texttt{add} never reads existing memories or their stored source and revision",
    ),
    NativeRow(
        "lightrag",
        None,
        "guo2025lightrag",
        "delete purges the document's chunks, vectors and entity and relation contributions",
    ),
    NativeRow(
        "hipporag",
        None,
        "gutierrez2025hipporag2",
        "delete removes the passage and the triples and phrases no other passage supports",
    ),
    NativeRow(
        "cognee",
        "Cognee",
        "arxiv2505_24478",
        "delete purges the item's chunks, summaries and the entities and edges only it produced",
    ),
)
# citations of the audited systems that have a paper (products and Khoj, Honcho, Supermemory have
# none); the native systems cite through NATIVE_ROWS
CITES = {
    "Graphiti / Zep": "arxiv2501_13956",
    "Letta (.af)": "packer2023memgpt",
    "Opal": "arxiv2604_02522",
    "User as Code": "arxiv2606_16707",
    "Solid pod": "mansour2016solid",
    "Plain RAG": "lewis2020rag",
}
# (title, audit groups) of the row groups; the native systems the audit does not cover are memory
# libraries
GROUPS = (
    ("products", ("vendor", "vendor (developer-side)", "platform")),
    ("memory libraries", ("memory layer",)),
    ("research systems and data stores", ("paper system", "personal data store")),
    ("our baselines and cells", ("baseline", "ours")),
)
NATIVE_GROUP = "memory libraries"
# the name a row prints where it differs from the audit's short name
NAMES = {"Plain RAG": "plain RAG", "zero system (ours)": "no-store BM25"}
BASIS_WORDS = {
    "code": "code",
    "paper": "paper",
    "documented": "documentation",
    "inferred": "indirect evidence",
}
NOTE_MARK = r"$^{a}$"
COLUMNS = 5


def _tex(s: str) -> str:
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        s = s.replace(a, b)
    return s


def placement(value: str) -> str:
    """A D or B placement in words ('---' where the system stores no derived item)."""
    return "---" if value == "N/A" else _tex(value)


def basis(row: dict[str, Any]) -> str:
    """What the D and B placements rest on: the evidence kinds of `D_basis` and `B_basis` in the
    order code, paper, documentation, indirect evidence, each once (a qualifier such as 'low
    confidence' is kept)."""
    found: dict[str, str] = {}
    for field in ("D_basis", "B_basis"):
        for part in str(row[field]).split("+"):
            kind, _, qualifier = part.strip().partition(" (")
            word = BASIS_WORDS.get(kind, _tex(kind))
            found.setdefault(word, f"{word} ({qualifier}" if qualifier else word)
    order = list(BASIS_WORDS.values())
    words = sorted(found, key=lambda w: order.index(w) if w in order else len(order))
    return ", ".join(found[w] for w in words)


def _label(d: Any, b: Any) -> tuple[str, str]:
    return str(d), str(b)


def _name(short_name: str, cite: str | None) -> str:
    name = NAMES.get(short_name, short_name)
    if name.startswith("HoldMem"):
        return r"\holdmem{}" + _tex(name.removeprefix("HoldMem").strip())
    name = _tex(name).replace("HippoRAG 2", "HippoRAG~2")
    return rf"{name} \citep{{{cite}}}" if cite else name


class Row(NamedTuple):
    group: str
    sort: str  # alphabetical within an external group; audit order within ours
    cells: tuple[str, ...]


def rows_of(audit: list[dict[str, Any]]) -> tuple[list[Row], list[str]]:
    """The table rows by group and the notes (the measured label of a native system whose audit
    placement differs from its documented label)."""
    group_of = {g: title for title, groups in GROUPS for g in groups}
    natives = {n.audit_name: n for n in NATIVE_ROWS if n.audit_name}
    tags = {n.key: n.tag for n in NATIVES}
    labels = {n.key: n.label for n in NATIVES}
    out: list[Row] = []
    notes: list[str] = []
    for i, r in enumerate(audit):
        if r["group"] == EXTRA:
            continue
        title = group_of[r["group"]]
        nat = natives.get(r["short_name"])
        d, b = placement(r["D"]), placement(r["B"])
        if nat is not None and _label(r["D"], r["B"]) != _label(*DOCUMENTED_LABELS[nat.key]):
            b += NOTE_MARK
            notes.append(
                rf"{NOTE_MARK}Measured at document level, as the native-system experiment runs"
                rf" {_tex(labels[nat.key])}: \Nat{tags[nat.key]}Label{{}}."
            )
        cite = nat.cite if nat else CITES.get(r["short_name"])
        name = _name(r["short_name"], cite)
        sort = f"{i:03d}" if title == GROUPS[-1][0] else name.lower()
        cells = (name, d, b, r"\checkmark" if nat else "", nat.reason if nat else basis(r))
        out.append(Row(title, sort, cells))
    covered = {n.key for n in NATIVE_ROWS if n.audit_name}
    for n in NATIVE_ROWS:
        if n.key in covered:
            continue
        d, b = (str(x) for x in DOCUMENTED_LABELS[n.key])
        name = _name(labels[n.key], n.cite)
        out.append(Row(NATIVE_GROUP, name.lower(), (name, d, b, r"\checkmark", n.reason)))
    return out, notes


def table(audit: list[dict[str, Any]]) -> str:
    rows, notes = rows_of(audit)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Memory systems on the $(D,B)$ grid: the audited systems, the further systems"
        r" that we run natively, and our cells."
        r" \checkmark: run natively in the native-system experiment. ---: the system stores no"
        r" derived item. Basis: what the placement rests on, and for a system we run, the property"
        r" of its code that sets $B$.}",
        r"\label{tab:audit}\label{tab:native-labels}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l l >{\raggedright\arraybackslash}p{1.6cm} c"
        r" >{\raggedright\arraybackslash}p{6.9cm}@{}}",
        r"\toprule",
        r"system & $D$ & $B$ & \begin{tabular}[b]{@{}c@{}}run\\natively\end{tabular} & basis \\",
    ]
    for title, _ in GROUPS:
        group = sorted((r for r in rows if r.group == title), key=lambda r: r.sort)
        if not group:
            continue
        lines += [r"\midrule", rf"\multicolumn{{{COLUMNS}}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        lines += [" & ".join(r.cells) + r" \\" for r in group]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines += [r"\par\smallskip", r"\parbox{\linewidth}{\footnotesize " + " ".join(notes) + "}"]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    rows = json.loads((root / "audit" / "system_audit.json").read_text())
    print("% generated by memshift.analysis.audit_table; do not edit")
    print(table(rows), end="")


if __name__ == "__main__":
    main()
