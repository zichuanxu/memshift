"""Paper numbers and table of the offline key audit (`analysis.e1_keys`). The audit uses no
model: the source-epoch key, the oracle key and the scope key on definition items are functions of
the world, so what they detect is known before any run. The source is the staged audit named
`keyaudit` in results/sources.json (`analysis/e1_keys.json`, worlds 1000-1019, branches src / def /
orphan); a stage that is not there renders every macro and table cell as the MISSING marker.

Every rate is computed from the counts the audit stores (not from its rounded rates), with half-up
rounding of the exact ratio, so a printed value is the value of the audit to the digit shown.

Macro names (letters only; `<K>` Source, Oracle (the `gold` key), Scope; `<B>` Src, Def, Orphan;
`<C>` All, Definition, Rule (structured fields), Free (free text); the scope key audits definition
items only, so it has the classes All (the same items) and Definition):

  KeyAuditWorlds                      number of worlds audited
  KeyAuditChanged<B>, KeyAuditUnchanged<B>   item-checkpoints whose gold differs from / equals the
                                      gold at t0, all items (the same for every key)
  KeyAuditMissed<K><B>                share of changed item-checkpoints the key had not invalidated
  KeyAuditMissedMax                   the largest such share over keys and branches
  KeyAuditFalse<K><B><C>              share of unchanged item-checkpoints the key had invalidated
  KeyAuditRederiv<K><B><C>            invalidations over the four checkpoints (one writer request
                                      each), summed over the worlds
  KeyAuditRederivWorld<K><B><C>       the same per world
  KeyAuditDefShare<K><B>              invalidations as a share of the definition item-checkpoints
                                      (worlds x definition items x four checkpoints)
  KeyAuditDefShare<K>Min / Max        the range of the key's share over the branches
  KeyAuditDefCheckpoints              definition item-checkpoints per branch

tab_keyaudit        appendix table: the changed and unchanged item-times per branch, false
                    invalidation by class, re-derivations per world, the single-field
                    re-derivations summed over the worlds (quoted in the text) and the
                    definition-item share, per branch and key
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, stage_json
from memshift.analysis.paper_numbers import _macro
from memshift.analysis.paper_tables import _row, notes

KEYS = (("source", "Source"), ("gold", "Oracle"), ("scope", "Scope"))
CLASS_TAGS = (
    ("all", "All"),
    ("definition", "Definition"),
    ("rule", "Rule"),
    ("norule", "Free"),
)
FIELDS = ("changed", "unchanged", "changed_detected", "unchanged_detected", "rederivations")


def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any] | None:
    """The staged audit (None where it is not staged)."""
    return stage_json(root, dict(sources), "keyaudit", "e1_keys.json")


def _half_up(num: int, den: int, scale: int, digits: int) -> str:
    """`scale * num / den` to `digits` places, rounded half up from the exact ratio."""
    q = Decimal(1).scaleb(-digits)
    return str((Decimal(scale * num) / Decimal(den)).quantize(q, rounding=ROUND_HALF_UP))


def _pct(num: int, den: int) -> str:
    return _half_up(num, den, 100, 1) + "\\%" if den else MISSING


def _counts(view: Mapping[str, Any] | None, branch: str, key: str, cls: str) -> dict[str, int]:
    """The counts of one (branch, key, class); `all` sums the classes the key audits."""
    by_class = (((view or {}).get(branch) or {}).get(key)) or {}
    rows = by_class.values() if cls == "all" else [by_class.get(cls)]
    total = dict.fromkeys(FIELDS, 0)
    for row in rows:
        if row is None:
            return {}
        for f in FIELDS:
            total[f] += row[f]
    return total if by_class else {}


def _worlds(view: Mapping[str, Any] | None) -> int:
    return len((view or {}).get("seeds") or [])


class _Out(dict[str, str]):
    """Macros of this block: a name defined twice is an error."""

    def put(self, *parts: str, value: str) -> None:
        name = "".join(("KeyAudit", *parts))
        if name in self:
            raise ValueError(f"macro {name} defined twice")
        self[name] = value


def _missed(c: dict[str, int]) -> tuple[int, int]:
    """Changed item-checkpoints the key had not invalidated, and all changed ones."""
    return (c["changed"] - c["changed_detected"], c["changed"]) if c else (0, 0)


def _false(c: dict[str, int]) -> tuple[int, int]:
    return (c["unchanged_detected"], c["unchanged"]) if c else (0, 0)


def _share(c: dict[str, int]) -> tuple[int, int]:
    """Invalidations over the item-checkpoints (definition items: four per item and world)."""
    return (c["rederivations"], c["changed"] + c["unchanged"]) if c else (0, 0)


def _count(c: dict[str, int], field: str) -> str:
    return f"{c[field]:,}" if c else MISSING


def _per_world(c: dict[str, int], worlds: int) -> str:
    return _half_up(c["rederivations"], worlds, 1, 1) if c and worlds else MISSING


def numbers(view: Mapping[str, Any] | None) -> dict[str, str]:
    """Every `KeyAudit` macro, from `load`."""
    out = _Out()
    n = _worlds(view)
    out.put("Worlds", value=str(n) if n else MISSING)
    missed: list[tuple[int, int]] = []
    shares: dict[str, list[tuple[int, int]]] = {k: [] for k, _ in KEYS}
    for b, bt in BRANCH_TAGS:
        source = _counts(view, b, "source", "all")
        definition = _counts(view, b, "gold", "definition")
        out.put("Changed", bt, value=_count(source, "changed"))
        out.put("Unchanged", bt, value=_count(source, "unchanged"))
        out.put(
            "DefCheckpoints",
            bt,
            value=f"{definition['changed'] + definition['unchanged']:,}" if definition else MISSING,
        )
        for k, kt in KEYS:
            allc = _counts(view, b, k, "all")
            missed.append(_missed(allc))
            out.put("Missed", kt, bt, value=_pct(*_missed(allc)))
            share = _share(_counts(view, b, k, "definition"))
            out.put("DefShare", kt, bt, value=_pct(*share))
            shares[k].append(share)
            for c, ct in CLASS_TAGS[:2] if k == "scope" else CLASS_TAGS:
                cc = _counts(view, b, k, c)
                out.put("False", kt, bt, ct, value=_pct(*_false(cc)))
                out.put("Rederiv", kt, bt, ct, value=_count(cc, "rederivations"))
                out.put("RederivWorld", kt, bt, ct, value=_per_world(cc, n))
    for k, kt in KEYS:
        known = [s for s in shares[k] if s[1]]
        for name, pick in (("Min", min), ("Max", max)):
            value = _pct(*pick(known, key=lambda s: s[0] / s[1])) if known else MISSING
            out.put("DefShare", kt, name, value=value)
    known = [s for s in missed if s[1]]
    out.put("MissedMax", value=_pct(*max(known, key=lambda s: s[0] / s[1])) if known else MISSING)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ------------------------------------------------------------------------------------- table
CAPTION = (
    r"What each key invalidates on the test worlds, audited without a model. Counted since the "
    r"fork, no key missed a changed item-time (the largest share not invalidated is {missed})."
)
NOTES = (
    r"Re-derivations: invalidations over the four checkpoints, one writer request each. Def.\ "
    r"share: share of the {cp} definition item-times of a shift that were re-derived. ---: the "
    r"scope key is built for definition items only."
)
KEY_LABELS = {"source": "source key", "gold": "oracle", "scope": "scope"}
BRANCH_LABELS = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}
DASH = r"\mtna"  # a class the scope key does not audit
# the value columns, each centred in a box as wide as its widest value: the item-times; false
# invalidation by class; re-derivations per world (all, definition items) and summed over the
# worlds (single-field items); the definition-item share
COLUMNS = (
    r"@{}l N{0,000}N{0,000} l N{100.0}@{\hspace{11pt}}N{100.0}@{\hspace{11pt}}N{100.0}"
    r"@{\hspace{11pt}}N{100.0} @{\hspace{14pt}} N{000.0}N{00.0}N{0,000} @{\hspace{14pt}} "
    r"N{00.0}@{}"
)


def _bare(value: str) -> str:
    return value.replace("\\%", "")


def _head(name: str, second: str, last: bool = False) -> str:
    """A two-line leaf header over a value column (the second line muted)."""
    return rf"\multicolumn{{1}}{{c{'@{}' if last else ''}}}{{\mthead{{{name}}}{{{second}}}}}"


def tab_keyaudit(view: Mapping[str, Any] | None) -> str:
    """The audit per shift and key: the shift's changed and unchanged item-times (the same for
    every key), false invalidation by class, re-derivations per world and, for single-field
    items, summed over the worlds, and the share of definition item-times re-derived."""
    m = numbers(view)
    caption = CAPTION.format(missed=m["KeyAuditMissedMax"])
    pct = r"(\%)"
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        r"\label{tab:keyaudit}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{{COLUMNS}}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Item-times} & & \multicolumn{4}{c}{Unchanged item-times "
        r"invalidated} & \multicolumn{3}{c}{Re-derivations} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-8}\cmidrule(lr){9-11}",
        _row(
            "Shift",
            r"\multicolumn{1}{c}{Changed}",
            r"\multicolumn{1}{c}{Unchanged}",
            "Key",
            _head("All", pct),
            _head("Def.", pct),
            _head("Field", pct),
            _head("Free", pct),
            _head("All", "per world"),
            _head("Def.", "per world"),
            _head("Field", f"{m['KeyAuditWorlds']} worlds"),
            _head(r"Def.\ share", pct, last=True),
        ),
        r"\midrule",
    ]
    for b, bt in BRANCH_TAGS:
        for j, (k, kt) in enumerate(KEYS):
            if k == "scope":  # definition items only
                fi = [DASH, m[f"KeyAuditFalse{kt}{bt}Definition"], DASH, DASH]
                rd = [DASH, m[f"KeyAuditRederivWorld{kt}{bt}Definition"], DASH]
            else:
                fi = [m[f"KeyAuditFalse{kt}{bt}{c}"] for _, c in CLASS_TAGS]
                rd = [m[f"KeyAuditRederivWorld{kt}{bt}{c}"] for c in ("All", "Definition")]
                rd.append(m[f"KeyAuditRederiv{kt}{bt}Rule"])  # summed over the worlds
            share = m[f"KeyAuditDefShare{kt}{bt}"]
            # the shift and its item-times once, on the shift's first line
            first = [BRANCH_LABELS[b], m[f"KeyAuditChanged{bt}"], m[f"KeyAuditUnchanged{bt}"]]
            lead = first if j == 0 else ["", "", ""]
            lines.append(_row(*lead, KEY_LABELS[k], *map(_bare, fi), *rd, _bare(share)))
        lines.append(r"\mtgap")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{mttabular}",
        notes(NOTES.format(cp=m["KeyAuditDefCheckpointsSrc"])),
        r"\end{table}",
    ]
    return "\n".join(lines)


def tables(view: Mapping[str, Any] | None) -> dict[str, str]:
    return {"tab_keyaudit.tex": tab_keyaudit(view)}
