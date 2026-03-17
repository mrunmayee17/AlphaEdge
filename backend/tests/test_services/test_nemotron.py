"""Test Nemotron LLM client with real API calls."""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_health_check(nemotron_client):
    assert await nemotron_client.health_check() is True


@pytest.mark.asyncio
async def test_simple_completion(nemotron_client):
    resp = await nemotron_client.chat_completion(
        [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        stream=False,
        max_tokens=10,
    )
    assert resp is not None
    assert "4" in resp


@pytest.mark.asyncio
async def test_json_completion(nemotron_client):
    resp = await nemotron_client.chat_completion_json([
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": 'Return {"answer": 42}'},
    ])
    assert isinstance(resp, dict)
    assert resp["answer"] == 42


@pytest.mark.asyncio
async def test_json_does_not_mutate_messages(nemotron_client):
    messages = [
        {"role": "system", "content": "Return valid JSON."},
        {"role": "user", "content": 'Return {"ok": true}'},
    ]
    original_len = len(messages)
    await nemotron_client.chat_completion_json(messages)
    assert len(messages) == original_len, "chat_completion_json should not mutate input"
