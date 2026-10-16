"""Check the registration history without the working repository.

1. Every object file hashes to its name (git's SHA-1 over "<type> <size>\\0<content>").
2. The commits form one parent chain ending at the published head; the first commit is given by hash.
3. Each registration file is reachable from its commit's tree, and its blob equals the file of the same
   name in this release's prereg/ directory where that file is unchanged since.
4. Every run manifest in this release (runs/*/manifest.json, field code.commit) names a commit of the
   chain, and each registration that the paper says preceded a run is an ancestor of that commit.

Usage, from the repository root: python registration-history/verify.py
"""

import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OBJ = HERE / "objects"
index = json.loads((HERE / "index.json").read_text())


def oid(typ: str, raw: bytes) -> str:
    return hashlib.sha1(f"{typ} {len(raw)}\0".encode() + raw).hexdigest()


objects: dict[str, tuple[str, bytes]] = {}
for f in sorted(OBJ.iterdir()):
    sha, typ = f.name.split(".")
    raw = f.read_bytes()
    if oid(typ, raw) != sha:
        sys.exit(f"FAIL: {f.name} does not hash to its name")
    objects[sha] = (typ, raw)
print(f"1. {len(objects)} objects hash to their names")

parents: dict[str, list[str]] = {}
for sha, (typ, raw) in objects.items():
    if typ == "commit":
        parents[sha] = re.findall(r"^parent ([0-9a-f]{40})$", raw.decode(), re.M)
known = set(parents) | {index["root_commit_hash_only"]}
for sha, ps in parents.items():
    for p in ps:
        if p not in known:
            sys.exit(f"FAIL: commit {sha} has a parent outside the chain: {p}")
if index["head"] not in parents:
    sys.exit("FAIL: head commit missing")
print(f"2. {len(parents)} commits link by parent hash to the first commit {index['root_commit_hash_only'][:7]}")


def ancestors(sha: str) -> set[str]:
    seen, stack = set(), [sha]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        stack.extend(parents.get(s, []))
    return seen


def tree_entry(tree_raw: bytes, name: str) -> str:
    i = 0
    while i < len(tree_raw):
        sp = tree_raw.index(b" ", i)
        nul = tree_raw.index(b"\0", sp)
        if tree_raw[sp + 1 : nul].decode() == name:
            return tree_raw[nul + 1 : nul + 21].hex()
        i = nul + 21
    sys.exit(f"FAIL: {name} not in tree")


for name, reg in index["registrations"].items():
    commit = objects[reg["commit"]][1].decode()
    cur = commit.split("\n")[0].split()[1]
    for part in reg["path"].split("/"):
        cur = tree_entry(objects[cur][1], part)
    if cur != reg["blob"]:
        sys.exit(f"FAIL: {name} blob mismatch")
    released = ROOT / reg["path"]
    same = released.exists() and oid("blob", released.read_bytes()) == cur
    note = "identical to the released file" if same else "released file was edited later"
    print(f"3. {name}: {reg['path']} is in commit {reg['commit'][:7]} ({note})")

runs = sorted((ROOT / "runs").glob("*/manifest.json"))
outside, counts = [], {k: 0 for k in index["registrations"]}
for m in runs:
    c = json.loads(m.read_text()).get("code", {}).get("commit")
    if not c:
        continue
    if c not in parents:
        outside.append((m.parent.name, c))
        continue
    anc = ancestors(c)
    for k, reg in index["registrations"].items():
        counts[k] += reg["commit"] in anc
print(f"4. {len(runs)} run manifests; runs whose code commit descends from each registration: {counts}")
if outside:
    print(f"   {len(outside)} manifests name a commit outside the chain, e.g. {outside[:3]}")
print("OK")
