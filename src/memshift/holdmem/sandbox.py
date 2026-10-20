"""Restricted interpreter for the model-written derivations of the D1s cell.

A program is Python source that defines `derive(records, as_of)`. It is parsed with `ast` and
rejected unless every node is on a whitelist: no imports, no names starting with `__`, no
attributes starting with `_`, no frame or generator introspection attributes, no wall clock
(`now`, `today`, `utcnow`: the only clock is `as_of`, so a program cannot depend on when it runs),
no bare `except`, and no `open`, `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `getattr`,
`setattr`, `delattr`, `input`, `breakpoint`. A single leading underscore is allowed on local
names (`_` as a throwaway loop variable is harmless); the rule is about double underscores.

Execution happens in a separate interpreter (`python -S -P` on this very file, hash seed fixed)
with a whitelisted `__builtins__` plus `datetime`, `timedelta` and `parse_ts`. (The builtins carry
an `__import__` that resolves only `_strptime` and `time`, which the C code of `datetime.strptime`
and `strftime` needs; a program cannot name it.) The child arms a
timer of 2 s of its own CPU time around the program (CPU time, not wall time, so that a busy host
cannot make a program time out that finishes on an idle one); the
timer's exception derives from BaseException, so `except Exception` inside a program cannot
swallow it. The parent additionally kills the child after a wall-clock grace period, which covers
a long uninterruptible C call. The address space of the child is capped
where the platform enforces it. Records travel as JSON on stdin, the answer comes back as JSON on
stdout. Only the standard library is imported here because the file doubles as the child's script.

Retraction-aware mode (`run_source_ra`, the `_ra` cells of E1): the program is
`derive(records, as_of, prior_reads)` and returns {"value": ..., "reads": [object ids]}. The child
wraps every record so that reading a content field of a record whose id the program does not list
in `reads` is an error ("undeclared read"); `object_id`, `source`, `revision`, `updated`, `cluster`
and the source's locator fields (`LOCATOR_FIELDS`: the fields that identify a record, e.g. a
contact's name) may be read on any record, so that a program can find the record it needs without
declaring every record it looked at. Whether the program also returns no value when a prior read is
gone is its own duty (the prompt asks for it); the parent only measures it (`program.py`).
"""

from __future__ import annotations

import ast
import builtins
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TIMEOUT_S = 2.0
KILL_GRACE_S = 60.0  # wall time: process start, JSON decoding, calls that ignore the timer, load
MAX_CODE_CHARS = 20_000
MEMORY_LIMIT_BYTES = 4 << 30

BUILTIN_FUNCTIONS = (
    "len",
    "sum",
    "min",
    "max",
    "sorted",
    "set",
    "list",
    "dict",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "abs",
    "any",
    "all",
    "enumerate",
    "range",
    "zip",
    "round",
    "isinstance",
    "reversed",
    "filter",
    "map",
)
EXCEPTION_CLASSES = (
    "Exception",
    "ValueError",
    "KeyError",
    "IndexError",
    "TypeError",
    "AttributeError",
    "ZeroDivisionError",
)
FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
        "input",
        "breakpoint",
    }
)
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "cr_frame",
        "cr_code",
        "cr_await",
        "ag_frame",
        "ag_code",
        "ag_await",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_back",
        "f_code",
        "tb_frame",
        "tb_next",
        "mro",
        "now",
        "today",
        "utcnow",
    }
)
# `ast.operator`, `boolop`, `unaryop`, `cmpop` and `expr_context` cover every operator and context.
ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Delete,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.For,
    ast.While,
    ast.If,
    ast.Raise,
    ast.Try,
    ast.ExceptHandler,
    ast.Assert,
    ast.Expr,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.BoolOp,
    ast.NamedExpr,
    ast.BinOp,
    ast.UnaryOp,
    ast.Lambda,
    ast.IfExp,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Compare,
    ast.Call,
    ast.keyword,
    ast.FormattedValue,
    ast.JoinedStr,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Starred,
    ast.Name,
    ast.List,
    ast.Tuple,
    ast.Slice,
    ast.operator,
    ast.boolop,
    ast.unaryop,
    ast.cmpop,
    ast.expr_context,
)
# Fields readable on any record in retraction-aware mode, besides the content fields a program has
# to declare (see the module docstring); the prompt of the `_ra` cells lists them from here.
METADATA_FIELDS = ("object_id", "source", "revision", "updated", "cluster")
LOCATOR_FIELDS: dict[str, tuple[str, ...]] = {
    "contacts": ("name",),
    "calendar": ("kind", "title", "appt_type"),
    "tasks": ("title", "project"),
    "email": ("from_addr",),
    "messages": ("conversation_with", "contact"),
    "notes": ("title",),
}
_TS_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d")
_CALLABLE_IMPORTS = frozenset({"_strptime", "time"})  # see `_restricted_import`
# `from datetime import datetime, timedelta` (any subset, no other alias) only binds names the
# program already has; it becomes `pass` before the whitelist check.
# Every other import statement stays disallowed syntax.
PROVIDED_IMPORTS: dict[str, frozenset[str]] = {"datetime": frozenset({"datetime", "timedelta"})}


class _DropProvidedImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        names = PROVIDED_IMPORTS.get(node.module or "")
        if (
            names is not None
            and node.level == 0
            and all(a.name in names and a.asname in (None, a.name) for a in node.names)
        ):
            return ast.copy_location(ast.Pass(), node)
        return node


def parse_program(code: str) -> ast.Module:
    """The program's syntax tree, with imports of provided names replaced by `pass`."""
    tree = _DropProvidedImports().visit(ast.parse(code))
    assert isinstance(tree, ast.Module)
    return ast.fix_missing_locations(tree)


def parse_ts(s: str) -> datetime:
    """Parse "YYYY-MM-DD HH:MM" (the format of every record time field and of `as_of`), also
    "YYYY-MM-DDTHH:MM" (`memshift.gen.model.parse_ts`) and a bare date (midnight)."""
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp: {s!r}")


def check_source(code: str, ra: bool = False) -> str | None:
    """None when `code` passes the whitelist and defines `derive(records, as_of)` (retraction-aware
    mode: `derive(records, as_of, prior_reads)`), else why not."""
    if len(code) > MAX_CODE_CHARS:
        return f"program too long ({len(code)} characters)"
    try:
        tree = parse_program(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return f"disallowed syntax: {type(node).__name__}"
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_NAMES):
            return f"forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES
        ):
            return f"forbidden attribute: {node.attr}"
        if isinstance(node, ast.FunctionDef) and node.name.startswith("__"):
            return f"forbidden name: {node.name}"
        if isinstance(node, ast.arg) and node.arg.startswith("__"):
            return f"forbidden name: {node.arg}"
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            return "disallowed syntax: bare except"
    derives = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "derive"]
    if not derives:
        return "no top-level function derive"
    a = derives[-1].args
    if len(a.args) != (3 if ra else 2) or a.posonlyargs or a.vararg or a.kwonlyargs or a.kwarg:
        return (
            "derive must take exactly (records, as_of, prior_reads)"
            if ra
            else ("derive must take exactly (records, as_of)")
        )
    return None


def _spawn(header: dict[str, Any], records_json: str, timeout: float) -> tuple[Any, str]:
    """Run the child on the header and the records: (its answer, "") or (None, why not)."""
    payload = (json.dumps(header) + "\n" + records_json).encode("ascii")
    with _SLOTS:
        try:
            proc = subprocess.run(
                [sys.executable, "-S", "-P", str(Path(__file__))],
                input=payload,
                capture_output=True,
                timeout=timeout + KILL_GRACE_S,
                env={**os.environ, "PYTHONHASHSEED": "0"},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, "timeout"
    try:
        return json.loads(proc.stdout.decode("ascii")), ""
    except ValueError:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
        return None, f"sandbox crashed (exit {proc.returncode}): {' '.join(tail)}"[:300]


def run_source(
    code: str, records_json: str, as_of: str, timeout: float = TIMEOUT_S
) -> tuple[bool, Any, str]:
    """Run `derive(records, as_of)` in the child process; (ok, raw value, error). `records_json`
    is the JSON text of the record list. The value is only checked to be a string, a list or
    None; typing against the item's value type is `program.validate_value`."""
    error = check_source(code)
    if error:
        return False, None, error
    answer, error = _spawn(
        {"code": code, "as_of": as_of, "timeout": timeout}, records_json, timeout
    )
    if answer is None:
        return False, None, error
    if answer["ok"]:
        return True, answer["value"], ""
    return False, None, str(answer["error"])


def run_source_ra(
    code: str,
    records_json: str,
    as_of: str,
    prior_reads: list[str],
    timeout: float = TIMEOUT_S,
) -> tuple[bool, Any, list[str], str]:
    """Run `derive(records, as_of, prior_reads)` in retraction-aware mode: (ok, raw value, the
    declared reads, error). A program that returns anything but {"value", "reads"}, or reads a
    content field of a record it did not declare, is not ok."""
    error = check_source(code, ra=True)
    if error:
        return False, None, [], error
    header = {"code": code, "as_of": as_of, "timeout": timeout, "prior_reads": prior_reads}
    answer, error = _spawn(header, records_json, timeout)
    if answer is None:
        return False, None, [], error
    if answer["ok"]:
        return True, answer["value"], answer["reads"], ""
    return False, None, [], str(answer["error"])


# Concurrent children are bounded so that CPU contention cannot turn a fast program into a timeout.
_SLOTS = threading.BoundedSemaphore(max(2, min(8, os.cpu_count() or 2)))


# ---------------------------------------------------------------------------- child process
class _Timeout(BaseException):
    """Raised by the timer; BaseException so that `except Exception` in a program passes it on."""


def _on_alarm(signum: int, frame: Any) -> None:
    raise _Timeout()


def _restricted_import(
    name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0
) -> Any:
    """The `__import__` of the program's builtins. Programs cannot reach it (no import statement,
    no `__` names), but the C code of `datetime.strptime` and `strftime` imports `_strptime` and
    `time` through the calling frame's builtins, so these two must resolve."""
    if name not in _CALLABLE_IMPORTS or level:
        raise ImportError(f"import of {name} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _limit_memory() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass  # not enforced on every platform; the parent's kill is the backstop


class _Reads:
    """The ids of the records whose content fields a retraction-aware program has read."""

    def __init__(self) -> None:
        self.ids: set[str] = set()


class _Tracked(dict):
    """A record that reports reads of its content fields (see the module docstring). Iterating
    over the keys, `in` and `keys()` only show which fields exist and are free."""

    def __init__(self, data: dict[str, Any], reads: _Reads):
        super().__init__(data)
        self._reads = reads
        self._id = str(data.get("object_id"))
        self._free = frozenset(METADATA_FIELDS + LOCATOR_FIELDS.get(str(data.get("source")), ()))

    def _touch(self, key: Any) -> None:
        if key not in self._free:
            self._reads.ids.add(self._id)

    def __getitem__(self, key: Any) -> Any:
        self._touch(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._touch(key)
        return super().get(key, default)

    def items(self) -> Any:
        self._reads.ids.add(self._id)
        return super().items()

    def values(self) -> Any:
        self._reads.ids.add(self._id)
        return super().values()

    def copy(self) -> Any:
        self._reads.ids.add(self._id)
        return dict(super().items())

    def __iter__(self) -> Any:
        # overriding iteration makes `dict(r)` and `{**r}` copy through __getitem__, not the table
        return super().__iter__()


def _ra_answer(result: Any, reads: _Reads) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != {"value", "reads"}:
        return {
            "ok": False,
            "error": "type mismatch: derive must return {'value': ..., 'reads': [...]}",
        }
    value, declared = result["value"], result["reads"]
    if not isinstance(declared, list) or not all(isinstance(i, str) for i in declared):
        return {"ok": False, "error": "type mismatch: reads must be a list of object ids"}
    if value is not None and not isinstance(value, (str, list)):
        return {"ok": False, "error": f"type mismatch: value is a {type(value).__name__}"}
    undeclared = sorted(reads.ids - set(declared))
    if undeclared:
        return {"ok": False, "error": f"undeclared read: {undeclared[0]}"}
    return {"ok": True, "value": value, "reads": declared}


def _run(
    code: str,
    records: Any,
    as_of: str,
    timeout: float,
    prior_reads: list[str] | None = None,
) -> dict[str, Any]:
    ra = prior_reads is not None
    error = check_source(code, ra=ra)
    if error:
        return {"ok": False, "error": error}
    program = compile(parse_program(code), "<program>", "exec")
    safe = {n: getattr(builtins, n) for n in (*BUILTIN_FUNCTIONS, *EXCEPTION_CLASSES)}
    namespace: dict[str, Any] = {
        "__builtins__": {**safe, "__import__": _restricted_import},
        "datetime": datetime,
        "timedelta": timedelta,
        "parse_ts": parse_ts,
    }
    _limit_memory()
    reads = _Reads()
    if prior_reads is not None:
        records = [_Tracked(r, reads) for r in records]
    signal.signal(signal.SIGPROF, _on_alarm)
    signal.setitimer(signal.ITIMER_PROF, timeout, 0.25)
    try:
        exec(program, namespace)
        if prior_reads is None:
            value = namespace["derive"](records, as_of)
        else:
            value = namespace["derive"](records, as_of, list(prior_reads))
    except _Timeout:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    if prior_reads is not None:
        return _ra_answer(value, reads)
    if value is not None and not isinstance(value, (str, list)):
        return {"ok": False, "error": f"type mismatch: derive returned {type(value).__name__}"}
    return {"ok": True, "value": value}


def _child() -> None:
    header = json.loads(sys.stdin.buffer.readline().decode("ascii"))
    records = json.loads(sys.stdin.buffer.read().decode("ascii"))
    answer = _run(
        header["code"],
        records,
        header["as_of"],
        float(header["timeout"]),
        header.get("prior_reads"),
    )
    try:
        text = json.dumps(answer)
    except (TypeError, ValueError):
        text = json.dumps({"ok": False, "error": "type mismatch: value is not plain text"})
    sys.stdout.buffer.write(text.encode("ascii"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    _child()
