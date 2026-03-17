"""NVIDIA Nemotron 120B client — OpenAI-compatible API wrapper."""

import json
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class NemotronClient:
    """Wraps OpenAI-compatible calls to NVIDIA Nemotron 120B API.

    Key: enable_thinking=False for all calls — otherwise response goes
    to reasoning_content field instead of content.
    """

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def chat_completion(
        self,
        messages: list[dict],
        stream: bool = True,
        max_tokens: int = 3000,
        temperature: float = 0.7,
    ) -> Any:
        """Completion call. Returns stream or string based on `stream` param."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=stream,
        )
        if stream:
            return resp  # caller iterates async
        return resp.choices[0].message.content

    async def chat_completion_stream(
        self,
        messages: list[dict],
        max_tokens: int = 3000,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Yields content chunks for WebSocket streaming to frontend."""
        stream = await self.chat_completion(
            messages, stream=True, max_tokens=max_tokens, temperature=temperature
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def chat_completion_json(
        self,
        messages: list[dict],
        max_tokens: int = 3000,
    ) -> dict:
        """Get JSON response, validate, retry on parse failure (max 2x).

        IMPORTANT: copies messages to avoid mutating the caller's list.
        """
        messages = [m.copy() for m in messages]  # never mutate caller's list
        for attempt in range(3):
            content = await self.chat_completion(
                messages, stream=False, max_tokens=max_tokens, temperature=0.3
            )
            if content is None:
                raise RuntimeError("Nemotron returned None content — check enable_thinking config")
            try:
                # Handle case where model wraps JSON in markdown code block
                cleaned = content.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                return json.loads(cleaned.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"Invalid JSON: {e}. Return ONLY valid JSON, no markdown, no explanation.",
                    })
                else:
                    raise ValueError(f"Failed to get valid JSON after 3 attempts. Last content: {content[:200]}")

    async def health_check(self) -> bool:
        """Quick ping to verify API key and model are working."""
        try:
            resp = await self.chat_completion(
                [{"role": "user", "content": "Reply with exactly: ok"}],
                stream=False,
                max_tokens=5,
                temperature=0.0,
            )
            return resp is not None and len(resp) > 0
        except Exception as e:
            logger.error(f"Nemotron health check failed: {e}")
            return False
