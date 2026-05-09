from agent_learning.models import (
    AgentCard,
    BudgetState,
    HookEvent,
    Message,
    ToolCall,
)


def test_message_to_openai_with_tool_calls():
    msg = Message(
        role="assistant",
        content="hi",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "x.txt"})],
    )
    payload = msg.to_openai()
    assert payload["role"] == "assistant"
    assert payload["content"] == "hi"
    assert payload["tool_calls"][0]["function"]["name"] == "read_file"
    assert "path" in payload["tool_calls"][0]["function"]["arguments"]


def test_budget_remaining_floors_at_zero():
    b = BudgetState(run_budget_usd=1.0, spent_usd=2.5)
    assert b.remaining_usd == 0.0


def test_agent_card_defaults():
    a = AgentCard(name="x")
    assert a.tools == ["*"]
    assert a.subagents == []
    assert HookEvent.PRE_TOOL_USE.value == "PreToolUse"
