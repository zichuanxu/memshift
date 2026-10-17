"""The round-3 swap phase (scripts/r3_swap/): the job files ask for what prereg v4 G-b, U5 and H-b
register (as amended 2026-10-16) with e1's own flags, and the phase driver serves one model at a
time, pre-warms before U5, runs every job once, never passes a Claude key and refuses to leave a
phase with unfinished jobs. The driver runs here against stand-ins for docker, the model servers
and scripts/run.sh."""

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from memshift.experiments.e1 import check_program_temperature, parse_cells, parse_seeds
from memshift.experiments.program_prewarm import build_parser

ROOT = Path(__file__).resolve().parents[1]
SWAP = ROOT / "scripts" / "r3_swap"
E1 = "memshift.experiments.e1"
PREWARM = "memshift.experiments.program_prewarm"
FAMILIES = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-4-12B-it",
    "gptoss": "openai/gpt-oss-20b",
}
# a G-b line of the Qwen program sets as queued in the Qwen phase (the open-weight lines mirror it)
QWEN_GB = (
    "600|memshift.experiments.e1|gb_qwen_p0_1000-1003|--seeds 1000-1003 --name "
    "e1_gb_qwen_p0_s1000-1003 --parallel-worlds 4 --branches src,def,orphan --discriminative "
    "--no-baselines --program-model Qwen/Qwen3.5-9B --program-temperature 0.7 --program-sample 0 "
    "--cells D1sB0,D1sB1_source --post-probes definition"
)
SWITCHES = {"--no-baselines", "--discriminative", "--probe-unnamed-defs", "--program-user-words"}
pytestmark = pytest.mark.skipif(not SWAP.is_dir(), reason="scripts/r3_swap is not released")


def lines(name: str) -> list[str]:
    text = (SWAP / name).read_text()
    return [x for x in text.splitlines() if x.strip() and not x.startswith("#")]


def args_of(line: str) -> dict[str, str]:
    thr, _, _, args = line.split("|")
    assert int(thr) > 0
    tokens = args.split()
    out: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        assert flag.startswith("--"), line
        if flag in SWITCHES:
            out[flag] = ""
            i += 1
        else:
            out[flag] = tokens[i + 1]
            i += 2
    return out


def e1_flags() -> set[str]:
    tree = ast.parse((ROOT / "src" / "memshift" / "experiments" / "e1.py").read_text())
    return {
        str(node.args[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_argument"
        and isinstance(node.args[0], ast.Constant)
    }


def seeds_cover(chunks: list[str], lo: int, hi: int) -> bool:
    got = [s for c in chunks for s in parse_seeds(c)]
    return sorted(got) == list(range(lo, hi + 1))


def all_jobs() -> list[str]:
    return [x for f in sorted(SWAP.glob("*.txt")) for x in lines(f.name)]


def test_job_lines_are_unique_and_use_e1s_flags():
    jobs = all_jobs()
    tags = [x.split("|")[2] for x in jobs]
    # pre-warms, U5, G-b open-weight replays, H-b, and the re-runs from the response cache:
    # 102 program sets and G-a, 6 stage-4 M2 runs, 1 main-study D_m run
    assert len(tags) == len(set(tags)) == 3 * 3 + 3 * 5 + 45 + 5 + 102 + 6 + 1
    flags = e1_flags()
    for line in jobs:
        module = line.split("|")[1]
        a = args_of(line)
        if module == E1:
            assert set(a) <= flags, line
            parse_cells(a["--cells"])
            assert parse_seeds(a["--seeds"])
        else:
            assert module == PREWARM, line
            build_parser().parse_args(line.split("|")[3].split())


def test_gb_open_weight_replays_mirror_the_qwen_lines():
    jobs = lines("jobs_gb_open.txt")
    assert len(jobs) == 3 * 3 * 5
    expected = []
    for fam, model in FAMILIES.items():
        for i in range(3):
            chunks = ["1000-1003", "1004-1007", "1008-1011", "1012-1015", "1016-1019"]
            assert seeds_cover(chunks, 1000, 1019)
            for c in chunks:
                expected.append(
                    QWEN_GB.replace("qwen_p0", f"{fam}_p{i}")
                    .replace("Qwen/Qwen3.5-9B", model)
                    .replace("--program-sample 0", f"--program-sample {i}")
                    .replace("1000-1003", c)
                )
    assert jobs == expected


def test_prewarms_cover_each_replayed_program_set():
    replayed = {
        (a["--program-model"], float(a["--program-temperature"]), int(a["--program-sample"]))
        for a in map(args_of, lines("jobs_gb_open.txt"))
    }
    warmed = set()
    for fam, model in FAMILIES.items():
        for line in lines(f"prewarm_{fam}.txt"):
            a = build_parser().parse_args(line.split("|")[3].split())
            assert a.program_model == model
            assert parse_seeds(a.seeds) == list(range(1000, 1020))
            assert a.branches == "src,def,orphan" and not a.program_user_words
            check_program_temperature(a.program_model, a.program_temperature)
            warmed.add((a.program_model, a.program_temperature, a.program_sample))
    assert warmed == replayed and len(warmed) == 9


def test_u5_and_hb_jobs():
    for fam, model in FAMILIES.items():
        jobs = [args_of(x) for x in lines(f"jobs_u5_{fam}.txt")]
        assert seeds_cover([a["--seeds"] for a in jobs], 1000, 1009)
        for a in jobs:
            assert a["--writer"] == model and "--reader" not in a  # reader = writer
            assert a["--cells"] == "D0B0,D0B0_refresh,D1B0_refresh,D0B0_upsert,D1B0_upsert"
            assert a["--post-probes"] == "definition" and a["--branches"] == "src,def,orphan"
            assert "--discriminative" in a and "--no-baselines" in a
            assert not any(k.startswith("--program") for k in a)
    hb = [args_of(x) for x in lines("jobs_hb.txt")]
    assert seeds_cover([a["--seeds"] for a in hb], 1000, 1009)
    for a in hb:
        assert a["--think-max-tokens"] == "32768"
        assert a["--cells"] == "D0B1_source,D0thinkB1_source"
        assert "--post-probes" not in a and "--writer" not in a and "--reader" not in a


def swap_endpoints() -> str:
    m = re.search(r"^SWAP_ENDPOINTS='(.*)'$", (SWAP / "swap_lib.sh").read_text(), re.M)
    assert m
    return m.group(1)


def test_swap_settings_match_the_job_files_and_stage4():
    lib = (SWAP / "swap_lib.sh").read_text()
    endpoints = json.loads(swap_endpoints())
    stage4 = (ROOT / "scripts" / "serve_stage4.sh").read_text()
    for fam, model in FAMILIES.items():
        assert endpoints[model] == "http://127.0.0.1:8004/v1"
        assert f"{fam}) echo {model} ;;" in lib
        assert f"  {model})" in stage4  # its pinned revision and engine flags
    assert endpoints["Qwen/Qwen3.5-9B"] == "http://127.0.0.1:8001/v1"


@pytest.mark.parametrize("script", sorted(p.name for p in SWAP.glob("*.sh")))
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(SWAP / script)], check=True)


# ---------------------------------------------------------------------------- the driver
STUBS = {
    "docker": r"""
echo "docker $*" >> "$FAKE/calls.log"
case "$1" in
  ps) if [[ "$*" == *purpose=vllm* ]]; then cat "$FAKE/vllm"; else cat "$FAKE/runs"; fi ;;
  rm) shift; [ "$1" = -f ] && shift
      for n in "$@"; do
        grep -vx "$n" "$FAKE/vllm" > "$FAKE/vllm.new"; mv "$FAKE/vllm.new" "$FAKE/vllm"
      done
      rm -f "$FAKE"/serving_* ;;
esac
""",
    "curl": r"""
url=${@: -1}
port=$(sed -E 's#^(http://)?localhost:([0-9]+)/.*#\2#' <<<"$url")
case $url in
  */health) [ -f "$FAKE/serving_$port" ] ;;
  */v1/models) [ -f "$FAKE/serving_$port" ] || exit 7
    read -r m l < "$FAKE/serving_$port"
    [ -f "$FAKE/no_len" ] && l=null  # a server that does not report its context
    echo "{\"object\":\"list\",\"data\":[{\"id\":\"$m\",\"max_model_len\":$l}]}" ;;
  */metrics) echo 'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0' ;;
esac
""",
    "pgrep": 'cat "$FAKE/pgrep"; [ -s "$FAKE/pgrep" ]',
    "nvidia-smi": "echo 0",
    "setsid": 'exec "$@"',
    "sleep": "exit 0",
}
SERVE = r"""
echo "$(basename "$0") $* MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS MEM_UTIL=$MEM_UTIL" \
  >> "$FAKE/calls.log"
echo "${MEMSHIFT_PREFIX}_$(tr '/.A-Z' '__a-z' <<<"$1")_202610170000" >> "$FAKE/vllm"
port=$3; [ "$(basename "$0")" = serve_model.sh ] && port=$4
echo "$1 ${MAX_LEN:-16384}" > "$FAKE/serving_$port"
mkdir -p logs/serving  # what serve_model.sh records
echo "{\"max_model_len\": ${MAX_LEN:-16384}}" > "logs/serving/$(tr '/.A-Z' '__a-z' <<<"$1").json"
"""
PREFIX = "ms_test"  # the container-name prefix of the host env file
QWEN_SERVERS = [
    f"{PREFIX}_qwen_qwen3_5-9b_202610160000",
    f"{PREFIX}_qwen_qwen3-embedding_202610160000",
]
RUN = r"""
echo "run $RUN_TAG $RUN_MODULE key=${CLAUDE_API_KEY:-none} endpoints=$MEMSHIFT_ENDPOINTS" \
  >> "$FAKE/calls.log"
echo "${MEMSHIFT_PREFIX}_run_x"
echo "exit=0"
"""


class Server:
    """A copy of logs/r3 with the swap files, stand-ins for the commands they call, and state."""

    def __init__(self, tmp: Path):
        self.root, self.fake, stubs = tmp / "memshift", tmp / "fake", tmp / "bin"
        self.r3 = self.root / "logs" / "r3"
        for d in (self.r3, self.fake, stubs, self.root / "scripts"):
            d.mkdir(parents=True)
        for f in SWAP.iterdir():
            (self.r3 / f.name).write_bytes(f.read_bytes())
        for name, body in STUBS.items():
            (stubs / name).write_text("#!/bin/bash\n" + body)
        (stubs / "python3").symlink_to(sys.executable)
        for name, body in {
            "serve_stage4.sh": SERVE,
            "serve_model.sh": SERVE,
            "run.sh": RUN,
        }.items():
            (self.root / "scripts" / name).write_text("#!/bin/bash\n" + body)
        for p in [*stubs.iterdir(), *(self.root / "scripts").iterdir()]:
            p.chmod(0o755)
        (self.root / ".env.test").write_text(
            f"export MEMSHIFT_PREFIX={PREFIX}\n"
            'export MEMSHIFT_ENDPOINTS=\'{"Qwen/Qwen3.5-9B": "http://127.0.0.1:8001/v1"}\'\n'
        )
        # the Qwen phase: its two servers, someone else's container, one finished job
        (self.fake / "vllm").write_text("\n".join([*QWEN_SERVERS, "other_vllm_x", ""]))
        (self.fake / "serving_8001").write_text("Qwen/Qwen3.5-9B 32768\n")
        (self.fake / "serving_8011").write_text("Qwen/Qwen3-Embedding-0.6B 8192\n")
        for state in ("runs", "pgrep", "calls.log"):
            (self.fake / state).write_text("")
        (self.r3 / "resume_sources.txt").write_text("# job file key\njobs_q.txt -\n")
        (self.r3 / "jobs_q.txt").write_text("600|memshift.experiments.e1|q_a|--seeds 1\n")
        self.done("q_a")
        self.env = {
            **os.environ,
            "PATH": f"{stubs}:/usr/bin:/bin",
            "MEMSHIFT_ROOT": str(self.root),
            "MEMSHIFT_ENV_FILE": ".env.test",
            "HOME": str(tmp),
            "FAKE": str(self.fake),
            "USER": "tester",
            "CLAUDE_API_KEY": "a-key-that-must-not-reach-any-run",
        }

    def done(self, tag: str) -> None:
        (self.r3 / f"{tag}.log").write_text(f"{PREFIX}_run_x\nexit=0\n")

    def phase(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.r3 / "phase.sh"), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=120,
        )

    def calls(self, prefix: str) -> list[str]:
        return [
            x for x in (self.fake / "calls.log").read_text().splitlines() if x.startswith(prefix)
        ]

    def wait_work(self, n: int) -> None:
        """Until n phase works have ended."""
        for _ in range(600):
            logs = [p.read_text() for p in self.r3.glob("phase_work_*.out")]
            if sum("work ended" in t for t in logs) >= n:
                return
            time.sleep(0.1)
        raise AssertionError(f"phase work did not end: {logs}")


def test_phase_driver(tmp_path):
    s = Server(tmp_path)

    # refusals: pre-warms not done; a Qwen job unfinished; a runner still up
    r = s.phase("qwen40k")
    assert r.returncode == 1 and "pre-warm" in r.stdout
    with (s.r3 / "jobs_q.txt").open("a") as f:
        f.write("600|memshift.experiments.e1|q_b|--seeds 2\n")
    r = s.phase("fam:llama")
    assert r.returncode == 1 and "phase qwen has 1 job(s) without exit=0" in r.stdout
    s.done("q_b")
    (s.fake / "pgrep").write_text("123 bash logs/r3/queue_runner.sh jobs.txt\n")
    r = s.phase("fam:llama", FORCE="1")
    assert r.returncode == 1 and "still running" in r.stdout
    (s.fake / "pgrep").write_text("")
    assert not (s.r3 / "phase.txt").exists() and s.calls("serve") == []

    # fam:llama: our two vLLM containers go, the family is served with its stage-4 settings (memory
    # 0.78 beside the embedding model), the three pre-warms run one after the other, then the five
    # U5 runs and the three re-run M2 runs, without a Claude key
    r = s.phase("fam:llama")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (s.r3 / "phase.txt").read_text() == "fam:llama\n"
    (rm,) = s.calls("docker rm")
    assert rm.split()[3:] == QWEN_SERVERS
    assert s.calls("serve") == [
        "serve_stage4.sh meta-llama/Llama-3.1-8B-Instruct 1 8004 MAX_LEN=16384 MAX_SEQS=48 "
        "MEM_UTIL=0.78",
        "serve_model.sh Qwen/Qwen3-Embedding-0.6B 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 1 8011 "
        "--runner pooling MAX_LEN=8192 MAX_SEQS=48 MEM_UTIL=0.12",
    ]
    s.wait_work(1)
    runs = s.calls("run ")
    tags = [x.split()[1] for x in runs]
    assert tags[:3] == ["gbw_llama_p0", "gbw_llama_p1", "gbw_llama_p2"]
    assert sorted(tags[3:]) == sorted(
        x.split("|")[2] for x in lines("jobs_u5_llama.txt") + lines("jobs_m2rerun_llama.txt")
    )
    assert all(f"key=none endpoints={swap_endpoints()}" in x for x in runs)
    assert all(x.split()[2] == PREWARM for x in runs[:3])
    assert "other_vllm_x" in (s.fake / "vllm").read_text()

    # again: already serving, every job done, nothing runs
    r = s.phase("fam:llama")
    assert r.returncode == 0 and "already serving" in r.stdout
    s.wait_work(2)
    assert len(s.calls("serve")) == 2 and len(s.calls("run ")) == 11
    status = s.phase("status").stdout
    assert re.search(r"fam:llama +prewarm_llama\.txt +done +3 +failed +0", status)
    assert re.search(r"fam:llama +jobs_u5_llama\.txt +done +5 ", status)
    assert re.search(r"qwen40k +jobs_gb_open\.txt +done +0 .* not started +45", status)

    # qwen40k (the other families' pre-warms done): Qwen with the registered context, H-b and the
    # G-b replays; a server that does not report its context is checked against logs/serving
    for fam in ("gemma", "gptoss"):
        for i in range(3):
            s.done(f"gbw_{fam}_p{i}")
    (s.fake / "no_len").write_text("")
    r = s.phase("qwen40k")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Qwen/Qwen3.5-9B serving on port 8001" in r.stdout
    max_len = re.search(r"^QWEN40K_MAX_LEN=(\d+)$", (SWAP / "swap_lib.sh").read_text(), re.M)
    assert max_len and int(max_len.group(1)) >= 40960
    assert s.calls("serve")[2] == (
        "serve_model.sh Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a 1 8001 "
        f"--language-model-only --reasoning-parser qwen3 MAX_LEN={max_len.group(1)} MAX_SEQS=128 "
        "MEM_UTIL=0.80"
    )
    s.wait_work(3)
    later = [x.split()[1] for x in s.calls("run ")[11:]]
    assert sorted(later) == sorted(
        x.split("|")[2]
        for x in lines("jobs_hb.txt")
        + lines("jobs_gb_open.txt")
        + lines("jobs_rerun_programs.txt")
        + lines("jobs_rerun_synth.txt")
    )
    r = s.phase("qwen40k")  # served with the right context: kept
    assert r.returncode == 0 and "already serving with context" in r.stdout
    s.wait_work(4)
    assert len(s.calls("serve")) == 3 and len(s.calls("run ")) == 11 + 50 + 102 + 1
