from agent_learning.store import JSONLStore


def test_jsonl_store_roundtrip(tmp_path):
    store = JSONLStore(tmp_path)
    run_id = store.start_run("inspect repo")
    store.record(run_id, kind="event", role="agent", payload={"message": "thinking"})
    store.record(run_id, kind="tool_result", role="read_file", payload={"content": "ok"})
    store.finish_run(run_id, final_message="done")

    events = store.load(run_id)
    kinds = [e.kind for e in events]
    assert kinds == ["run_started", "event", "tool_result", "run_finished"]
    assert events[1].payload["message"] == "thinking"
    assert run_id in store.list_runs()


def test_recent_text_picks_relevant_kinds(tmp_path):
    store = JSONLStore(tmp_path)
    run_id = store.start_run("g")
    store.record(run_id, kind="event", role="x", payload={"message": "alpha"})
    store.record(run_id, kind="usage", role="x", payload={"prompt_tokens": 1})
    store.record(run_id, kind="memory", role="x", payload={"message": "beta"})
    text = store.recent_text(run_id)
    assert "alpha" in text
    assert "beta" in text
    assert "prompt_tokens" not in text
