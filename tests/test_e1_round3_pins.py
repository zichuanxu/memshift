"""The round-3 flags (`--think-max-tokens`, `--defwriter`, `--program-model`,
`--program-temperature`, `--program-sample`, `--probe-unnamed-defs`) leave every existing request
unchanged at their defaults. The writer, thinking-writer and program requests of the existing cells
(as response-cache keys, which are SHA256 digests of the full request bodies), the items they write
and the probe questions are pinned on two small worlds; the digests are those of the code without
the round-3 flags."""

import pytest

from memshift.experiments.e1 import parse_cell
from memshift.probes import make_probes
from tests.e1_replay import digest, replay

TOKENS = (
    "D0B0",
    "D1B0",
    "D0B1_source",
    "D1B1_source",
    "D1sB0",
    "D1sB1_source",
    "D1sB0_ra",
    "D0thinkB0",
    "D0thinkB1_source",
    "D0fullB1_source",
    "D1B0_refresh",
)

PINS = {
    "def": {
        "requests": "3b28d66356b74c5c551c485f110648e9191cdd541059060343764dbaa89e3783",
        "think_requests": "9adbc7b58a64cf728b26869a90505861992155e553867a9e20c1761662e56bdd",
        "items": "ec3a1494f8397a0ceebac790045ed445a747c1e7ae643dc51f2d0eaa50e9e03b",
        "probes": "56323db9fac26289332bf8f3a166d31e7ae271f153f0305861d9a7b9d507d34a",
    },
    "src": {
        "requests": "dafb5fc8ed656940614cfd8d679d78809f1d9327f4189fe277f4d0b931e46288",
        "think_requests": "e2c210a3d5f4269dff7ebc871d1bce91baa9b7fe8eeccde5592b95b303dd0fe5",
        "items": "8d809aa95402e63ae7e88ce1d603a3bc32b5ebaa80f76721381b590168c9d797",
        "probes": "f1f1de25e3d8d6efd5a029990a1d0846efabe0fb5197f03449a633716109e6de",
    },
}


def snapshot(branch: str) -> dict[str, str]:
    """Digests of the requests, items and probe questions of the existing cells at the defaults."""
    rep = replay([parse_cell(t) for t in TOKENS], branch=branch, weeks=(1, 2))
    times = [rep.t0, rep.at(1), rep.at(2)]
    return {
        "requests": digest({name: sorted(llm.keys()) for name, llm in rep.llms.items()}),
        "think_requests": digest({n: sorted(llm.keys()) for n, llm in rep.think_llms.items()}),
        "items": digest(
            {n: [i.to_json() for i in hm.items.values()] for n, hm in rep.cells.items()}
        ),
        "probes": digest([p.question_text for p in make_probes(rep.world, rep.log, times)]),
    }


@pytest.mark.parametrize("branch", ["def", "src"])
def test_defaults_send_the_requests_of_the_code_before_round_3(branch):
    assert snapshot(branch) == PINS[branch]
