import threading
import time

from memshift.llm import LLMClient, LLMResult


def test_identical_concurrent_requests_post_once(tmp_path, monkeypatch):
    llm = LLMClient("http://unused|http://unused2", "m", tmp_path / "c.sqlite", max_workers=8)
    posts = []
    lock = threading.Lock()

    def fake_post(body, key):
        with lock:
            posts.append(key)
        time.sleep(0.2)
        return LLMResult('{"ok": 1}', 10, 2, 0.2, False, "stop", key)

    monkeypatch.setattr(llm, "_post", fake_post)
    req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    other = {"messages": [{"role": "user", "content": "bye"}], "max_tokens": 8}
    results = llm.map([req] * 6 + [other])
    assert len(posts) == 2
    assert sum(not r.cached for r in results) == 2
    assert llm.complete(**req).cached  # later calls hit the persistent cache
    assert llm.usage["prompt_tokens"] == 8 * 10  # accounting counts every logical call


def test_template_kwargs_only_for_models_with_a_chat_template(tmp_path):
    msgs = [{"role": "user", "content": "hi"}]
    qwen = LLMClient("http://unused", "Qwen/Qwen3.5-9B", tmp_path / "q.sqlite")
    body = qwen.request_body(msgs, 8, None)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}  # cache keys unchanged
    mistral = LLMClient("http://unused", "mistralai/Ministral-3-14B", tmp_path / "m.sqlite")
    assert "chat_template_kwargs" not in mistral.request_body(msgs, 8, None)


def test_stage4_families_reasoning_off(tmp_path):
    msgs = [{"role": "user", "content": "hi"}]
    body = {
        m: LLMClient("http://unused", m, tmp_path / f"{i}.sqlite").request_body(msgs, 8, None)
        for i, m in enumerate(
            (
                "meta-llama/Llama-3.1-8B-Instruct",
                "openai/gpt-oss-20b",
                "google/gemma-4-12B-it",
                "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
            )
        )
    }
    assert "chat_template_kwargs" not in body["meta-llama/Llama-3.1-8B-Instruct"]
    assert body["openai/gpt-oss-20b"]["reasoning_effort"] == "low"
    assert body["google/gemma-4-12B-it"]["chat_template_kwargs"] == {"enable_thinking": False}
    mimo = body["XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"]
    assert mimo["chat_template_kwargs"] == {"enable_thinking": False}  # cache keys unchanged
