"""The existing E1 cells are unchanged by the control cells: the writer and program requests they
send (as response-cache keys), the contexts they serve at t0 and after two checkpoints, and the
item rows they write are pinned on two small worlds. The digests were taken on the code before the
control cells were added; a change here means a cached run would no longer be a hit, or an old cell
behaves differently."""

from datetime import timedelta

import pytest

from memshift.experiments.common import world_params
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.probes import make_probes
from memshift.reader import context_hash
from tests.e1_replay import digest
from tests.fakes import RecordingLLM

EXISTING = [
    CellConfig(d, b, g) for d in (0, 1) for b, g in ((0, "source"), (1, "source"), (1, "cited"))
] + [CellConfig(1, 0, synth=True), CellConfig(1, 1, "source", synth=True)]

PINS = {
    "def": {
        "requests": "6a18f8d36d9115c96f3461fd68f661b938c988e65def4e9d5450931568dab8e9",
        "contexts": "d527b93adb410b4143ac863c70e24589120949efd4204715f09d85f387d41234",
        "items": "e7818cfc70b92db9e96337aa422ccf2f133a83806088faf6af4c2fe479ce5352",
    },
    "orphan": {
        "requests": "9d78ef67326fbe51789a4498a660bd610376e6c9e3a5fa8eae3aea334b61cb83",
        "contexts": "db04c7f0c5695089e1afbbb8846a562d517355131f1bfe835572f43c396a0335",
        "items": "8c43a646afb5e09c0bf3d66418a484de2b48cc960d10541c66cc516987c4a4da",
    },
}


@pytest.mark.parametrize("branch", ["def", "orphan"])
def test_existing_cells_send_the_same_requests_and_write_the_same_items(branch):
    world = generate_branch(5, branch, world_params(3))
    log = SourceLog(world.events)
    t0 = world.fork - timedelta(minutes=1)
    cells = {}
    llms = {}
    for c in EXISTING:
        llms[c.name] = RecordingLLM()
        cells[c.name] = HoldMem(world, log, c, llms[c.name], "fake/writer-a")
        cells[c.name].workers = 4
        cells[c.name].build(t0)
    contexts = {}
    for k in (0, 1, 2):
        t = t0 if k == 0 else world.fork + timedelta(weeks=k)
        if k:
            for hm in cells.values():
                hm.maintain(t)
        for name, hm in cells.items():
            for p in make_probes(world, log, [t]):
                contexts[f"{name}|{k}|{p.probe_id}"] = context_hash(hm.serve(p).blocks)
    requests = {name: sorted(llm.keys()) for name, llm in llms.items()}
    items = {name: [i.to_json() for i in hm.items.values()] for name, hm in cells.items()}
    assert digest(requests) == PINS[branch]["requests"]
    assert digest(contexts) == PINS[branch]["contexts"]
    assert digest(items) == PINS[branch]["items"]
    # the additive item field stays out of the rows of cells that record nothing in it
    assert not any("note" in row for rows in items.values() for row in rows)
