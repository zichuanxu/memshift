"""Two clients that share one response cache and post the same request at the same time.

Demonstrates the cause of the round-3 U5 anchor failure (fails on the current `ResponseCache.put`,
which is INSERT OR REPLACE). The committed stage-4 M2 runs (Llama-3.1-8B, gpt-oss-20b) and the E2
warm-up of the same family (`e2.warm`, natural branch at t0, same requests as every branch's t0
snapshot) ran at the same time on one cache file; both missed some t0 keys and both posted. The
server does not repeat itself for an identical request at temperature 0, so the replies differed;
each process used its own reply, and the cache kept the one stored last. The M2 run's src branch
(built first) used its own replies; its def and orphan branches, built hours later, and the U5
rerun read the warm-up's. Two worlds of one E1 process are the same case: `e1.run_world` builds its
own clients, so world-independent requests (D1s programs) are posted once per world.
"""

import threading

from memshift.llm import LLMClient, LLMResult


def test_concurrent_posters_of_one_request_get_the_reply_the_cache_keeps(tmp_path, monkeypatch):
    path = tmp_path / "c.sqlite"
    first = LLMClient("http://unused", "m", path)  # e.g. the stage-4 M2 run
    second = LLMClient("http://unused", "m", path)  # e.g. the E2 warm-up, another process
    both_missed = threading.Barrier(2)
    first_stored = threading.Event()

    def server(reply: str, wait: threading.Event | None):
        def post(body, key):
            both_missed.wait(5)  # both clients missed the cache and are posting
            if wait is not None:
                wait.wait(5)  # this reply arrives after the other one is stored
            return LLMResult(f'{{"reply": "{reply}"}}', 10, 2, 0.1, False, "stop", key)

        return post

    monkeypatch.setattr(first, "_post", server("first", None))
    monkeypatch.setattr(second, "_post", server("second", first_stored))
    req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    used: dict[str, str] = {}

    def call(name: str, llm: LLMClient) -> None:
        used[name] = llm.complete(**req).text
        if name == "first":
            first_stored.set()

    threads = [
        threading.Thread(target=call, args=a) for a in (("first", first), ("second", second))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert set(used) == {"first", "second"}
    # a later phase of the same run (the M2 def branch) reads back what that run used at t0
    assert first.complete(**req).text == used["first"]
    # a rerun (all cache hits, as U5) gets what the original run used
    rerun = LLMClient("http://unused", "m", path).complete(**req)
    assert rerun.cached
    assert rerun.text == used["first"] == used["second"]
