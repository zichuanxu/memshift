"""Paper numbers and table of the offline key audit (`analysis.e1_keys`; prereg/decisions.md
2026-09-30, "WS2 offline audit"). The audit uses no model: the source-epoch key, the oracle key and
the scope key on definition items are functions of the world, so what they detect is known before
any run. The source is the staged audit named `keyaudit` in results/sources.json
(`analysis/e1_keys.json`, worlds 1000-1019, branches src / def / orphan); a stage that is not there
renders every macro and table cell as the MISSING marker.

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

tab_keyaudit        appendix table: false invalidation by class, re-derivations per world, the
                    single-field re-derivations summed over the worlds (quoted in the text) and
                    the definition-item share, per branch and key
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, stage_json
from memshift.analysis.paper_numbers import _macro
from memshift.analysis.paper_tables import _row

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
    r"Offline audit of the binding keys on the {n} test worlds (no model): what each key "
    r"invalidates, from the worlds alone. Keys: the source-epoch key, the oracle key (built from "
    r"the gold dependencies) and the scope key (built from what each definition ranges over, "
    r"definition items only). False invalidation is the share of unchanged item-times (gold "
    r"value and status equal to those before the shift) that the key invalidated. It is given "
    r"for all items and for definition (def.), single-field (field) and free-text "
    r"(free) items. Re-derivations are invalidations over the four checkpoints, one writer "
    r"request each, per world for all and definition items, and in total over the {n} worlds "
    r"for single-field items. Def.\ share: the share of the {cp} definition item-times of a "
    r"branch that were re-derived. Item-times with a changed / unchanged gold value: $\Ssrc$ "
    r"{src_c} / {src_u}, $\Sdef$ {def_c} / {def_u}, $\Sdel$ {or_c} / {or_u}. No key missed a "
    r"changed item-time (the largest share not invalidated is {missed})."
)
KEY_LABELS = {"source": "source epoch", "gold": "oracle", "scope": "scope"}
BRANCH_LABELS = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}
DASH = "---"


def _bare(value: str) -> str:
    return value.replace("\\%", "")


def tab_keyaudit(view: Mapping[str, Any] | None) -> str:
    m = numbers(view)
    caption = CAPTION.format(
        n=m["KeyAuditWorlds"],
        cp=m["KeyAuditDefCheckpointsSrc"],
        src_c=m["KeyAuditChangedSrc"],
        src_u=m["KeyAuditUnchangedSrc"],
        def_c=m["KeyAuditChangedDef"],
        def_u=m["KeyAuditUnchangedDef"],
        or_c=m["KeyAuditChangedOrphan"],
        or_u=m["KeyAuditUnchangedOrphan"],
        missed=m["KeyAuditMissedMax"],
    )
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        r"\label{tab:keyaudit}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{@{}ll rrrr rr r r@{}}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{false invalidation (\%)} & "
        r"\multicolumn{3}{c}{re-derivations} & def. \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
        rf"& & & & & & \multicolumn{{2}}{{c}}{{per world}} & {m['KeyAuditWorlds']} worlds "
        r"& share \\",
        r"\cmidrule(lr){7-8}\cmidrule(lr){9-9}",
        _row("branch", "key", "all", "def.", "field", "free", "all", "def.", "field", r"(\%)"),
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
            label = BRANCH_LABELS[b] if j == 0 else ""
            lines.append(_row(label, KEY_LABELS[k], *map(_bare, fi), *rd, _bare(share)))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tables(view: Mapping[str, Any] | None) -> dict[str, str]:
    return {"tab_keyaudit.tex": tab_keyaudit(view)}
